import json
import argparse
import torch
import vllm
from datasets import load_dataset
from eval.truthfulqa.utilities import format_prompt


def create_prompt_with_tulu_chat_format(messages, bos="<s>", eos="</s>", add_bos=True):
    formatted_text = ""
    for message in messages:
        if message["role"] == "system":
            formatted_text += "<|system|>\n" + message["content"] + "\n"
        elif message["role"] == "user":
            formatted_text += "<|user|>\n" + message["content"] + "\n"
        elif message["role"] == "assistant":
            formatted_text += "<|assistant|>\n" + message["content"].strip() + eos + "\n"
        else:
            raise ValueError(
                "Tulu chat template only supports 'system', 'user' and 'assistant' roles. Invalid role: {}.".format(message["role"])
                )
    formatted_text += "<|assistant|>\n"
    formatted_text = bos + formatted_text if add_bos else formatted_text
    return formatted_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="gsm8k")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--use_chat_format", action="store_true")
    parser.add_argument("--num_completions", type=int, default=5)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    if args.dataset == "gsm8k":
        prompt_prefix = "Answer the following question.\n\nQuestion: " if args.use_chat_format else "Question: "
        prompt_suffix = " Answer:"
        dataset = load_dataset("gsm8k", "main", split="train")
        raw_prompts = [prompt_prefix + d["question"].strip() + prompt_suffix for d in dataset]
        stop_sequence = ["\n"]
        max_new_tokens = 512
    elif args.dataset == "truthful_qa":
        dataset = load_dataset("truthful_qa", "generation", split="validation")
        # Keeping it consistent with the logic in eval.truthful_qa.run_eval.run_hf_model
        raw_prompts = [
            format_prompt(d["question"], preset="qa", format="general") for d in dataset
        ]
        stop_sequence = ["\n\n"]
        max_new_tokens = 50
        if args.use_chat_format:
            raw_prompts = [
                prompt + "A:" if prompt[-1] in ["\n", " "] else " A:" for prompt in raw_prompts
            ]
    else:
        raise NotImplementedError(f"Cannot handle dataset {args.dataset}")

    if args.use_chat_format:
        formatted_prompts = []
        for prompt in raw_prompts:
            messages = [{"role": "user", "content": prompt}]
            # TODO: Assuming model expects Tulu chat format
            formatted_prompt = create_prompt_with_tulu_chat_format(messages, add_bos=False)
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


    outputs = model.generate(prompts, sampling_params)
    completions = [[it.outputs[i].text for i in range(args.num_completions)] for it in outputs]
    with open(args.output, "w") as outfile:
        for prompt, instance_completions in zip(raw_prompts, completions):
            print(
                json.dumps({"prompt": prompt, "completions": instance_completions}),
                file=outfile
            )


if __name__ == "__main__":
    main()
