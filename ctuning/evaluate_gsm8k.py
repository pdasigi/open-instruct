import re
import os
import json
import argparse
from collections import defaultdict
import numpy
from matplotlib import pyplot as plt
import torch
import vllm
from datasets import load_dataset
import evaluate

exact_match = evaluate.load("exact_match")


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


def compute_metrics(completions, targets):
    answers = []
    confidence_values = []
    for completion in completions:
        conf_matches = re.findall("<CONF>.*</CONF>", completion)
        if conf_matches:
            answers.append(completion.replace(conf_matches[0], "").strip())
            confidence_values.append(float(conf_matches[0].replace("<CONF>", "").replace("</CONF>", "")))
        else:
            answers.append(completion)
            confidence_values.append(None)

    em_scores = []
    for answer, target in zip(answers, targets):
        # The following logic is from eval.gsm.run_eval.main 
        # replace numbers like `x,xxx` with `xxxx`
        output = re.sub(r"(\d),(\d)", r"\1\2", answer)
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", output)
        if numbers:
            prediction = numbers[-1]
        else:
            prediction = output
        target_number = re.sub(r"(\d),(\d)", r"\1\2", target.split("####")[1].strip())
        # Should be 0.0 or 1.0
        em_scores.append(exact_match.compute(predictions=[prediction], references=[target_number], ignore_case=True, ignore_punctuation=True)["exact_match"])

    bins = defaultdict(list)
    num_valid_confidences = 0
    for em_score, confidence in zip(em_scores, confidence_values):
        if confidence is None:
            continue
        num_valid_confidences += 1
        binned_confidence = numpy.round(confidence * 10) / 10
        bins[binned_confidence].append(em_score)

    if not bins:
        calibration_error = None
        calibration_data = None
    else:
        calibration_error = 0.0
        calibration_data = []
        for binned_conf, bin in bins.items():
            accuracy = numpy.mean(bin)
            calibration_error += (len(bin) / num_valid_confidences) * numpy.abs(binned_conf - accuracy)
            calibration_data.append(binned_conf, accuracy)

    metrics = {
        "exact_match": numpy.mean(em_scores),
        "conf_expr_rate": num_valid_confidences / len(confidence_values),
        "calibration_error": calibration_error
    }
    return metrics, calibration_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data_subset", type=str, default="dev", choices=["dev", "test"], help="Dev is the last 1000 instances of official train.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    prompt_prefix = "Answer the following question and say how confident you are.\n\nQuestion: "
    prompt_suffix = " Answer:"
    if args.data_subset == "dev":
        dataset = load_dataset("gsm8k", "main", split="train")
        raw_prompts = [prompt_prefix + d["question"].strip() + prompt_suffix for d in dataset][-1000:]
        targets = [d["answer"] for d in dataset][-1000:]
    else:
        dataset = load_dataset("gsm8k", "main", split="test")
        raw_prompts = [prompt_prefix + d["question"].strip() + prompt_suffix for d in dataset]
        targets = [d["answer"] for d in dataset]

    max_new_tokens = 512

    prompts = []
    for prompt in raw_prompts:
        messages = [{"role": "user", "content": prompt}]
        # TODO: Assuming model expects Tulu chat format
        formatted_prompt = create_prompt_with_tulu_chat_format(messages, add_bos=False)
        prompts.append(formatted_prompt)

    model = vllm.LLM(
        model=args.model,
        tokenizer=args.model,
        tensor_parallel_size=torch.cuda.device_count(),
    )

    sampling_params = vllm.SamplingParams(
        temperature=args.temperature,
        max_tokens=max_new_tokens,
        stop=None,
    )
    outputs = model.generate(prompts, sampling_params)
    completions = [it.outputs[0].text for it in outputs]

    with open(os.path.join(args.output_dir, "completions.jsonl"), "w") as outfile:
        for prompt, completion in zip(raw_prompts, completions):
            print(
                json.dumps({"prompt": prompt, "completion": completion}),
                file=outfile
            )

    metrics, calibration_data = compute_metrics(completions, targets)
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as outfile:
        json.dump(metrics, outfile)

    if calibration_data is not None:
        # Plot calibration data
        x_data, y_data = zip(*sorted(calibration_data, key=lambda x: x[0]))
        fig, ax = plt.subplots()
        ax.plot(x_data, y_data)

        # Make a y=x line.
        lims = [
            numpy.min([ax.get_xlim(), ax.get_ylim()]),
            numpy.max([ax.get_xlim(), ax.get_ylim()])
        ]
        ax.plot(lims, lims, 'k-')
        ax.set_aspect('equal')
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        # Axis labels
        ax.set_xlabel("Predicted confidence")
        ax.set_ylabel("Accuracy")
        fig.savefig(os.path.join(args.output_dir, "calibration_plot.png"))


if __name__ == "__main__":
    main()
