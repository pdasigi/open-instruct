import copy
import subprocess
import yaml
import random
from datetime import date

today = date.today().strftime("%m%d%Y")

with open("configs/beaker_configs/vif_finetune.yaml", 'r') as f:
    default_yaml = f.read()
d1 = yaml.load(default_yaml, Loader=yaml.FullLoader)

# cluster = "ai2/general-cirrascale"
# cluster = "ai2/yizhongw-a100-80gb"
#cluster = "ai2/allennlp-cirrascale"
cluster = "ai2/jupiter-cirrascale-2"
num_gpus = 8
per_device_batch_size = 1
d1['tasks'][0]['context']['cluster'] = cluster
d1['tasks'][0]['context']['priority'] = "high"
d1['tasks'][0]['context']['preemptible'] = True # requried for Jupiter/Pluto
d1['tasks'][0]['resources']['gpuCount'] = num_gpus

# modify here for different set of experiments
experiment_group = "dataset_comparison"
wandb_project = "verifiable_if"


# ----------------------- dataset comparison -----------------------
if experiment_group == "dataset_comparison":
    datasets = [
        #"tulu_and_dae",
        "tulu_and_collie_and_dae",
        #"tulu",
        "tulu_and_collie_alpaca"
    ]
    model_size = "8B"

    for dataset in datasets:
        d = copy.deepcopy(d1)

        # name and description
        exp_name = f"oi_ft_llama3_{model_size}_tulu-colliev3_best-conf_{dataset}_{today}"
        d['description'] = exp_name
        d['tasks'][0]['name'] = exp_name

        # model specific
        for mount_dataset in d['tasks'][0]['datasets']:
            if mount_dataset["mountPath"] == "/hf_llama_models":
                mount_dataset["source"]["beaker"] = f"davidw/Meta-Llama-3-8B"
        d['tasks'][0]['arguments'][0] = d['tasks'][0]['arguments'][0].replace(
            "--per_device_train_batch_size 2", 
            f"--per_device_train_batch_size {per_device_batch_size}"
        )
        d['tasks'][0]['arguments'][0] = d['tasks'][0]['arguments'][0].replace(
            "--gradient_accumulation_steps 16",
            f"--gradient_accumulation_steps {128 // per_device_batch_size // num_gpus}"
        )
        d['tasks'][0]['arguments'][0] = d['tasks'][0]['arguments'][0].replace(
            "--num_processes 8",
            f"--num_processes {num_gpus}"
        )

        if model_size == "13B":
            d['tasks'][0]['arguments'][0] = d['tasks'][0]['arguments'][0].replace(
                "--deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf",
                "--deepspeed_config_file configs/ds_configs/stage3_offloading_accelerate.conf",
            )


        d['tasks'][0]['arguments'][0] = d['tasks'][0]['arguments'][0].replace(
            "--train_file /data/alpaca_data_original_template.jsonl", 
            f"--train_file /data/{dataset}_data.jsonl"
        )

        # wandb specific
        d['tasks'][0]['arguments'][0] = d['tasks'][0]['arguments'][0].replace(
            "--report_to tensorboard",
            "--report_to wandb"
        )
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
        # print(d)

        fn = "configs/beaker_configs/auto_created/{}.yaml".format(exp_name)
        file = open(fn, "w")
        yaml.dump(d, file, default_flow_style=True)
        file.close()

        cmd = "beaker experiment create {} --workspace ai2/tulu-3-dev".format(fn)
        subprocess.Popen(cmd, shell=True)
