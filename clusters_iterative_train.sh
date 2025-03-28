#!/bin/bash
#SBATCH -A imi@h100
#SBATCH -C h100
#SBATCH --cpus-per-task=24
#SBATCH --time=1:55:59
#SBATCH --gres=gpu:1
#SBATCH --array=0-399 # 1 (seeds) x 100 (dataset types) x 2 (ratios)
#SBATCH -o logs/clusters_iterative_train_%A_%a.log
#SBATCH -e logs/clusters_iterative_train_%A_%a.log
#SBATCH -J clusters_iterative_train

start_time=$(date +%s)

cat $0

echo ""

echo "SLURM_JOB_ID: "$SLURM_JOB_ID"_"$SLURM_ARRAY_TASK_ID | tee -a $log_path

# 6 ratios
#ratios=(0.0625 0.125 0.25 0.5 0.75 1)
ratios=(0.125 0.25) # 1/8, 1/4
ratios_len=${#ratios[@]}

# 4 dataset types
dattypes=(cluster_{0..199})
dattypes_len=${#dattypes[@]}

ratio_id=$((SLURM_ARRAY_TASK_ID % ratios_len))
dattype_id=$(((SLURM_ARRAY_TASK_ID / ratios_len) % dattypes_len))
seed_id=0

datetime=`date +"%Y-%m-%d_%H-%M-%S"`
datetime_nano=`date +"%Y-%m-%d_%H-%M-%S.%N"`
seed=${seed_id}_${datetime_nano}

dattype=${dattypes[$dattype_id]}
ratio=${ratios[$ratio_id]}

echo "dattype:"$dattype
echo "ratio:"$ratio
echo "seed:"$seed

# human dataset size for generation 0
per_participant_ft_dataset_size=4000

# Define ratio array to use for calculating the generated dataset size
per_participant_ai_dataset_size=$(echo "$ratio * $per_participant_ft_dataset_size / 1" | bc)
per_participant_human_dataset_size=$(( per_participant_ft_dataset_size - per_participant_ai_dataset_size ))

echo "ft_size:"$per_participant_ft_dataset_size
echo "generated_dataset_size:"$per_participant_ai_dataset_size
echo "human_dataset_size:"$per_participant_human_dataset_size

# mixed
mixed_models_options=(
  # llama 3.1 1B
  "/lustre/fsn1/projects/rech/imi/utu57ed/.cache/huggingface/hub/models--unsloth--Llama-3.2-1B/snapshots/1d05b8ce9cd75f6baca1ccebf9653626ac261438"
  # qwen 2.5 1.5b
  "/lustre/fsn1/projects/rech/imi/utu57ed/.cache/huggingface/hub/models--unsloth--Qwen2.5-1.5B/snapshots/8951671def651bbedbcdea3751f46cf35e78dfa9"
  # smolm (1.7B)
  "/lustre/fsn1/projects/rech/imi/utu57ed/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM-1.7B/snapshots/d7449ff7241c863f3e8accc475155f0f97afa011"
  # falcon (1b)
  "/lustre/fsn1/projects/rech/imi/utu57ed/.cache/huggingface/hub/models--tiiuae--Falcon3-1B-Base/snapshots/34183642457812e78b53466798c3a818485ac969"
)
model="mixed"

dataset_name="webis_reddit"
split="all"

source /linkhome/rech/genini01/utu57ed/.bashrc

module purge
module load arch/h100
module load python/3.11.5
conda activate unsloth_311

n_part=1
accumulate=1

epochs=1
rank=16
alpha=16
per_device_batch_size=16
lr=2e-4
lr_scheduler="linear"
warmup_ratio=0.00125 # 5/4000; 5 steps

temp=1.5
min_p=0.2
n_generations=20

exp_path=webis_clusters_results_v2/human_ai_ratio_dataset_${dataset_name}_type_${dattype}_participants_${n_part}/generated_${per_participant_ai_dataset_size}_human_${per_participant_human_dataset_size}_unsloth/seed_${seed}_${datetime}
#exp_path=test/human_ai_ratio_dataset_${dataset_name}_type_${dattype}_participants_${n_part}/generated_${per_participant_ai_dataset_size}_human_${per_participant_human_dataset_size}_unsloth/seed_${seed}_${datetime}

mkdir -p $exp_path
log_path=$exp_path/log.txt

# presample human datasets
python -u presample_human_datasets.py \
  --exp-path $exp_path --generations $n_generations --n-participants "$n_part" \
  --per-participant-human-dataset-size-gen-0 $per_participant_ft_dataset_size \
  --per-participant-human-dataset-size $per_participant_human_dataset_size \
  --human-dataset $dataset_name \
  --split $split \
  --dataset-type $dattype \
  --seed "${seed}" 2>&1 | tee -a $log_path

for gen_i in $(seq 0 $((n_generations - 1)))
do

  echo -e "\033[32mGEN: $gen_i\033[0m"
  # prepare the training datasets
  if [ "$gen_i" -eq 0 ]; then
    # in first the first generation
    current_per_participant_human_dataset_size=$per_participant_ft_dataset_size
    current_per_participant_ai_dataset_size=0

  else
    current_per_participant_human_dataset_size=$per_participant_human_dataset_size
    current_per_participant_ai_dataset_size=$per_participant_ai_dataset_size

  fi

  python -u sample_datasets.py \
    --exp-path $exp_path --generation "$gen_i" --n-participants "$n_part" \
    --per-participant-human-dataset-size $current_per_participant_human_dataset_size \
    --per-participant-ai-dataset-size $current_per_participant_ai_dataset_size \
    --human-dataset $dataset_name \
    --load-presampled-human-dataset \
    --split $split \
    --accumulate $accumulate \
    --dataset-type $dattype \
    --seed "${seed}_gen_${gen_i}" 2>&1 | tee -a $log_path

  for part_i in $(seq 0 $((n_part-1))); do
    echo -e "\033[32mPart: $part_i (gen:$gen_i)\033[0m"

    # Sample a random model if model == mixed
    selected_model=$(shuf -e "${mixed_models_options[@]}" -n 1)
    echo "Selected model "$selected_model

    save_dir=$exp_path"/gen_"$gen_i"/part_"$part_i

    generate_n=$per_participant_ai_dataset_size

    python -u ft_and_gen.py \
        --save-dir $save_dir \
        --seed "${seed}_gen_${gen_i}_part_${part_i}" \
        --model-name $selected_model \
        --generate \
        --epochs $epochs \
        --rank $rank \
        --alpha $alpha \
        --per-device-batch-size $per_device_batch_size \
        --lr $lr \
        --lr-scheduler $lr_scheduler \
        --warmup-ratio $warmup_ratio \
        --temp $temp \
        --min-p $min_p \
        --gen-n $generate_n \
        --dataset-name $dataset_name \
        --dataset-type $dattype \
        --split $split 2>&1 | tee -a $log_path

  done
done


conda activate eval_311
python evaluate_generations.py --emb --seed-dir $exp_path

end_time=$(date +%s)
# Calculate the difference
duration=$(( end_time - start_time ))

# Calculate hours, minutes, and seconds
hours=$(( duration / 3600 ))
minutes=$(( (duration % 3600) / 60 ))
seconds=$(( duration % 60 ))

# Format the output to hh:mm:ss
printf -v elapsed_time "%02d:%02d:%02d" $hours $minutes $seconds

echo "Elapsed time: $elapsed_time " 2>&1 | tee -a $log_path
