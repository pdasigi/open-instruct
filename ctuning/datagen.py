import re
import json
import argparse
import warnings
import torch
import vllm
from datasets import load_dataset
from transformers import AutoTokenizer
from huggingface_hub import login
import evaluate
import pandas as pd

import os
import sys
sys.path.insert(0, os.path.abspath(os.pardir))
from eval.truthfulqa.utilities import format_prompt
from eval.truthfulqa.metrics import run_hf_classifier_eval
from eval.utils import load_hf_lm_and_tokenizer


exact_match = evaluate.load("exact_match")
HF_TOKEN = os.environ.get('HF_TOKEN')
if HF_TOKEN is None:
    warnings.warn("HF_TOKEN is not set. This is needed for gated models.")
else:
    login(token=HF_TOKEN)


def estimate_confidence_gsm(target, completions):
    predictions = []
    for completion in completions:
        # The following logic is from eval.gsm.run_eval.main 
        # replace numbers like `x,xxx` with `xxxx`
        output = re.sub(r"(\d),(\d)", r"\1\2", completion)
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", output)
        if numbers:
            predictions.append(numbers[-1])
        else:
            predictions.append(output)
    target_number = re.sub(r"(\d),(\d)", r"\1\2", target.split("####")[1].strip())
    references = [target_number] * len(predictions)
    em_score = exact_match.compute(predictions=predictions, references=references, ignore_case=True, ignore_punctuation=True)["exact_match"]
    return em_score


def estimate_confidence_truthfulqa(pd_dataset, idx, num_completions):
    truth_info_accuracies = []
    for i in range(num_completions):
        truth_info_accuracies.append(pd_dataset.loc[idx, f"sample_{i} truth-info acc"])
    return sum(truth_info_accuracies) / num_completions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="gsm8k")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--use_chat_format", action="store_true")
    parser.add_argument("--num_completions", type=int, default=5)
    parser.add_argument("--hf_truth_model", type=str, default="allenai/truthfulqa-truth-judge-llama2-7B")
    parser.add_argument("--hf_info_model", type=str, default="allenai/truthfulqa-info-judge-llama2-7B")
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    if args.dataset == "gsm8k":
        prompt_prefix = "Answer the following question.\n\nQuestion: " if args.use_chat_format else "Question: "
        prompt_suffix = " Answer:"
        dataset = load_dataset("gsm8k", "main", split="train")
        raw_prompts = [prompt_prefix + d["question"].strip() + prompt_suffix for d in dataset]
        targets = [d["answer"] for d in dataset]
        stop_sequence = ["\n"]
        max_new_tokens = 512
    elif args.dataset == "truthful_qa":
        dataset = load_dataset("truthful_qa", "generation", split="validation")
        # Keeping it consistent with the logic in eval.truthful_qa.run_eval.run_hf_model
        raw_prompts = [
            format_prompt({"Question": d["question"]}, preset="qa", format="general") for d in dataset
        ]
        if args.use_chat_format:
            raw_prompts = [
                prompt + ("A:" if prompt[-1] in ["\n", " "] else " A:") for prompt in raw_prompts
            ]
        stop_sequence = ["\n\n"]
        max_new_tokens = 50
    else:
        raise NotImplementedError(f"Cannot handle dataset {args.dataset}")

    if args.use_chat_format:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        formatted_prompts = []
        for prompt in raw_prompts:
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
            formatted_prompts.append(formatted_prompt)
        prompts = formatted_prompts
    else:
        prompts = raw_prompts

    model = vllm.LLM(
        model=args.model,
        tokenizer=args.model,
        tensor_parallel_size=torch.cuda.device_count(),
    )

    sampling_params = vllm.SamplingParams(
        n=args.num_completions,
        temperature=args.temperature,
        max_tokens=max_new_tokens,
        stop=stop_sequence if not args.use_chat_format else None,
    )
    sampled_outputs = model.generate(prompts, sampling_params)
    sampled_completions = [[it.outputs[i].text for i in range(args.num_completions)] for it in sampled_outputs]

    greedy_params = vllm.SamplingParams(
        n=1,
        temperature=0.0,
        max_tokens=max_new_tokens,
        stop=stop_sequence if not args.use_chat_format else None,
    )
    greedy_outputs = model.generate(prompts, greedy_params)
    greedy_completions = [it.outputs[0].text for it in greedy_outputs]

    confidence_values = []
    if args.dataset == "gsm8k":
        for target, samp_comps in zip(targets, sampled_completions):
            confidence_values.append(estimate_confidence_gsm(target, samp_comps))

    elif args.dataset == "truthful_qa":
        # Saving memory
        del model
        # Need to create a pandas dataframe for evaluating truthfulness and informativeness
        pd_dataset = pd.DataFrame(dataset)
        pd_dataset = pd_dataset.rename(columns={"question": "Question"})
        for idx, completions in  zip(pd_dataset.index, sampled_completions):
            for i, completion in enumerate(completions):
                pd_dataset.loc[idx, f"sample_{i}"] = completion 

        print(f"Loading truth classifier from {args.hf_truth_model}")
        classifier, tokenizer = load_hf_lm_and_tokenizer(
            model_name_or_path=args.hf_truth_model, 
            tokenizer_name_or_path=args.hf_truth_model,
            device_map="balanced_low_0" if torch.cuda.device_count() > 1 else "auto",
        )
        print(f"Classifying truthfulness")
        for i in range(args.num_completions):
            pd_dataset = run_hf_classifier_eval(f"sample_{i}", 'truth', classifier, tokenizer, pd_dataset, info=False)

        print(f"Loading informativeness classifier from {args.hf_info_model}")
        classifier, tokenizer = load_hf_lm_and_tokenizer(
            model_name_or_path=args.hf_info_model, 
            tokenizer_name_or_path=args.hf_info_model,
            device_map="balanced_low_0" if torch.cuda.device_count() > 1 else "auto",
        )
        print(f"Classifying informativeness")
        for i in range(args.num_completions):
            pd_dataset = run_hf_classifier_eval(f"sample_{i}", 'info', classifier, tokenizer, pd_dataset, info=True)
            pd_dataset[f"sample_{i} truth-info acc"] = pd_dataset[f"sample_{i} truth acc"] * pd_dataset[f"sample_{i} info acc"]

        for idx in pd_dataset.index:
            confidence_values.append(estimate_confidence_truthfulqa(pd_dataset, idx, args.num_completions))
    else:
        raise NotImplementedError(f"Cannot handle dataset {args.dataset}")


    with open(os.path.join(args.output_dir, "completions.jsonl"), "w") as outfile:
        for prompt, samp_comps, greedy_comp, confidence in zip(raw_prompts, sampled_completions, greedy_completions, confidence_values):
            print(
                json.dumps(
                    {
                        "prompt": prompt,
                        "sampled_completions": samp_comps,
                        "greedy_completion": greedy_comp,
                        "confidence": confidence,
                    }),
                file=outfile
            )


    ctuning_dataset_name = f"{args.dataset.split('/')[-1]}_{args.model.split('/')[-1]}_temp{args.temperature}"
    with open(os.path.join(args.output_dir, f"{ctuning_dataset_name}_train.jsonl"), "w") as outfile:
        for i, (prompt, greedy_comp, confidence) in enumerate(zip(raw_prompts, greedy_completions, confidence_values)):
            if args.dataset == "gsm8k":
                prompt = prompt.replace("Answer the following question.", "Answer the following question and say how confident you are.", 1)
                response = greedy_comp + f"\n<CONF>{confidence}</CONF>"
            else:
                raise NotImplementedError
            output_datum = {
                "dataset": ctuning_dataset_name,
                "id": f"{ctuning_dataset_name}_{i}",
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response}
                ]
            }
            print(json.dumps(output_datum), file=outfile)


if __name__ == "__main__":
    main()
