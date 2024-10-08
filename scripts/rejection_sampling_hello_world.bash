#!/bin/bash

mkdir -p output/shards
num_prompts=400
num_shards=5
prompts_per_shard=$((num_prompts / num_shards))
shared_hf_repo_id=rejection_sampling_$RANDOM 
num_generations=5
generation_model=cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr
reward_model=cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr
sft_dataset=trl-internal-testing/tldr-preference-sft-trl-style
num_gpus=1
mkdir -p output/shards/$shared_hf_repo_id

# Prepare the command string
command=""

# Loop through shards
for ((i=0; i<num_shards; i++))
do
    # Calculate start and end indices for this shard
    start_idx=$((i * prompts_per_shard))
    end_idx=$(((i + 1) * prompts_per_shard))
    
    # Adjust the end index for the last shard to include any remaining prompts
    if [ $i -eq $((num_shards - 1)) ]; then
        end_idx=$num_prompts
    fi
    
    # Build the command string for this shard
    shard_command="python open_instruct/generation.py \
    --dataset_name $sft_dataset \
    --model_name_or_path $generation_model \
    --dataset_start_idx $start_idx \
    --dataset_end_idx $end_idx \
    --save_filename output/shards/$shared_hf_repo_id/$i.jsonl \
    --n $num_generations --tensor_parallel_size $num_gpus && \
    python open_instruct/rejection_sampling.py \
    --input_filename output/shards/$shared_hf_repo_id/$i.jsonl \
    --model_names_or_paths $reward_model \
    --save_filename output/shards/$shared_hf_repo_id/scores_$i.jsonl \
    --hf_repo_id $shared_hf_repo_id \
    --no_add_timestamp \
    --n $num_generations \
    --push_to_hub \
    --num_gpus $num_gpus && \
    echo Finished shard $((i+1)) of $num_shards"

    # Add the shard command to the main command string
    if [ -z "$command" ]; then
        command="$shard_command"
    else
        command="$command -- $shard_command"
    fi
done

echo $command

# Run the combined command
echo "Submitting all shards in one command"
python mason.py \
    --cluster ai2/general-cirrascale-a5000 ai2/allennlp-cirrascale ai2/general-cirrascale-a100-80g-ib \
    --budget ai2/allennlp \
    --priority low \
    --preemptible \
    --gpus $num_gpus -- $command

echo "All shards submitted"