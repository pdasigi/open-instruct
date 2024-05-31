import copy
import subprocess
import yaml
import random
from datetime import date

today = date.today().strftime("%m%d%Y")

with open("beaker_configs/default_ctuning.yaml", 'r') as f:
    default_yaml = f.read()
d1 = yaml.load(default_yaml, Loader=yaml.FullLoader)


use_lora = False
cluster = "ai2/allennlp-cirrascale"
num_gpus = 2
d1['tasks'][0]['context']['cluster'] = cluster
d1['tasks'][0]['context']['priority'] = "low"
d1['tasks'][0]['resources']['gpuCount'] = num_gpus

# modify here for different set of experiments
experiment_group = "ctuning"
wandb_project = "tulu-calibration"


# ----------------------- dataset comparison -----------------------
datasets = [
    "gsm8k_Mistral-7B-Instruct-v0.2_temp0.2",
    "gsm8k_Mistral-7B-Instruct-v0.2_temp0.4",
    "gsm8k_Mistral-7B-Instruct-v0.2_temp0.6",
    "gsm8k_Mistral-7B-Instruct-v0.2_temp0.8",
    "gsm8k_Mistral-7B-Instruct-v0.2_temp1.0",
]
beaker_model_path = None
hf_model_path = "mistralai/Mistral-7B-Instruct-v0.2"
assert (hf_model_path is None) != (beaker_model_path is None)

if use_lora:
    with open("beaker_configs/default_merge_lora.yaml", 'r') as f:
        merge_lora_yaml = f.read()
    d2 = yaml.load(merge_lora_yaml, Loader=yaml.FullLoader)
    d2['tasks'][0]['context']['priority'] = d1['tasks'][0]['context']['priority']

for dataset in datasets:
    d = copy.deepcopy(d1)

    # name and description
    exp_name = f"ctuning_{dataset}_{today}" if not use_lora else f"ctuning_lora_{dataset}_{today}"
    d['description'] = exp_name
    d['tasks'][0]['name'] = exp_name

    # use lora?
    if use_lora:
        d['tasks'][0]['arguments'][0] = d['tasks'][0]['arguments'][0].replace(
            "--output_dir /output/",
            "--output_dir /output/lora_weights/"
        )
        d['tasks'][0]['arguments'][0] += ' --use_lora --lora_rank 64 --lora_alpha 16 --lora_dropout 0.1'

    # number of gpus
    d['tasks'][0]['arguments'][0] = d['tasks'][0]['arguments'][0].replace(
        "--num_processes 4",
        f"--num_processes {num_gpus}"
    )

    # model specific
    if beaker_model_path is not None:
        mount_dataset = {'mountPath': '/model', 'source': {'beaker': beaker_model_path}}
        d['tasks'][0]['datasets'].append(mount_dataset)
    elif hf_model_path is not None:
        d['tasks'][0]['arguments'][0] = d['tasks'][0]['arguments'][0].replace(
            "--tokenizer_name /model",
            f"--tokenizer_name {hf_model_path}"
        ).replace(
            "--model_name_or_path /model",
            f"--model_name_or_path {hf_model_path}"
        )

    d['tasks'][0]['arguments'][0] = d['tasks'][0]['arguments'][0].replace(
        "--gradient_accumulation_steps 32",
        f"--gradient_accumulation_steps {128 // 2 // num_gpus}"
    )


    # dataset specific
    d['tasks'][0]['arguments'][0] = d['tasks'][0]['arguments'][0].replace(
        "--train_file /data/gsm8k_tulu2-7b_temp0.2_train.jsonl", 
        f"--train_file /data/{dataset}_train.jsonl"
    )

    # wandb specific
    for env in d['tasks'][0]['envVars']:
        if env['name'] == "WANDB_DISABLED":
            env['value'] = False
        if env['name'] == "WANDB_PROJECT":
            env['value'] = wandb_project
    d['tasks'][0]['envVars'].append({
        'name': 'WANDB_NAME', 'value': exp_name
    })
    d['tasks'][0]['envVars'].append({
        'name': 'WANDB_RUN_GROUP', 'value': experiment_group
    })

    if use_lora:
        lora_d = copy.deepcopy(d2)['tasks'][0]
        lora_d['name'] = f"{exp_name}_lora_merge"
        if beaker_model_path is not None:
            mount_dataset = {'mountPath': '/base_model', 'source': {'beaker': beaker_model_path}}
            lora_d['datasets'] = [mount_dataset]
        elif hf_model_path is not None:
            lora_d["arguments"] = [hf_model_path if argument == "/base_model" else argument for argument in lora_d["arguments"]]

        d['tasks'].append(lora_d)


    # print(d)

    fn = "beaker_configs/auto_created/{}.yaml".format(exp_name)
    file = open(fn, "w")
    yaml.dump(d, file, default_flow_style=True)
    file.close()

    cmd = "beaker experiment create {} --workspace ai2/pradeepd-open-instruct".format(fn)
    subprocess.Popen(cmd, shell=True)
