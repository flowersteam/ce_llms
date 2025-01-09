#!/bin/bash
#SBATCH -A imi@h100
#SBATCH -C h100
##SBATCH -A vgw@a100
##SBATCH -C a100
#SBATCH --cpus-per-task=24
#SBATCH --time=19:55:59
#SBATCH --gres=gpu:1
## all parts
#SBATCH --array=0-124
##parts 1 2 4
##SBATCH --array=0-14,25-39,50-64,75-89,100-114
##parts 10
##SBATCH --array=15-19,40-44,65-69,90-94,115-119
##parts 1 2 4 10
##SBATCH --array=0-19,25-44,50-69,75-94,100-119
## parts 20
##SBATCH --array=20-24,45-49,70-74,95-99,120-124
#SBATCH -o logs/scale_iterative_train_%A_%a.log
#SBATCH -e logs/scale_iterative_train_%A_%a.log
#SBATCH -J scale_iterative_train

start_time=$(date +%s)

#cat $0 # print this script
#echo ""
#echo "SLURM_JOB_ID: "$SLURM_JOB_ID"_"$SLURM_ARRAY_TASK_ID | tee -a $log_path

#############
# useful bash commands
#############
# average total time for generated_500_
# grep -l "generated_500_" logs/scale_iterative_train_1700708_* | xargs grep "Total time (ft_and_gen):" | sed 's/.*(\([0-9.]*\) secs).*/\1/' | awk '{sum+=$1} END {print "Average seconds:", sum/NR}'

# track done generations
#watch -c -n 1 eval "cat logs/scale_iterative_train_1700708_*  | grep "GEN: " | sort -V | uniq -c"

# simulate run
#for i in $(seq 15 19) ; do SLURM_ARRAY_TASK_ID=$i bash iterative_train_scale.sh ; done
#for i in $(seq 40 44) ; do SLURM_ARRAY_TASK_ID=$i bash iterative_train_scale.sh ; done
#for i in $(seq 65 59) ; do SLURM_ARRAY_TASK_ID=$i bash iterative_train_scale.sh ; done
#for i in $(seq 90 94) ; do SLURM_ARRAY_TASK_ID=$i bash iterative_train_scale.sh ; done
#for i in $(seq 115 119) ; do SLURM_ARRAY_TASK_ID=$i bash iterative_train_scale.sh ; done
#############

ratios=(0.125 0.25 0.5 0.75 1)
parts=(1 2 4 10 20)


ratio_len=${#ratios[@]}
parts_len=${#parts[@]}

seed_id=$((SLURM_ARRAY_TASK_ID / (ratio_len * parts_len)))
remainder=$((SLURM_ARRAY_TASK_ID % (ratio_len * parts_len)))
ratio_i=$((remainder % ratio_len))
part_i=$((remainder / ratio_len))

n_part=${parts[$part_i]} # Access the array element using ${}
ratio=${ratios[$ratio_i]} # Access the array element using ${}
echo "ratio: $ratio"
echo "n_part: $n_part"
echo "seed:"$seed_id

gen_train_ratio=0.1


datetime=`date +"%Y-%m-%d_%H-%M-%S"`
datetime_nano=`date +"%Y-%m-%d_%H-%M-%S.%N"`
seed=${seed_id}_${datetime_nano}

echo "ratio_id:"$ratio_id
echo "seed:"$seed

# human dataset size for generation 0
per_participant_ft_dataset_size=4000


# Define ratio array to use for calculating the generated dataset size
per_participant_ai_dataset_size=$(echo "$ratio * $per_participant_ft_dataset_size / 1" | bc)
per_participant_human_dataset_size=$(( per_participant_ft_dataset_size - per_participant_ai_dataset_size ))

echo "ft_size:"$per_participant_ft_dataset_size
echo "generated_dataset_size:"$per_participant_ai_dataset_size
echo "human_dataset_size:"$per_participant_human_dataset_size

generate_n=$per_participant_ai_dataset_size

if (( $(echo "$gen_train_ratio < 1" | bc -l) )); then
    # Calculate the first term: ceil(generate_n * gen_train_ratio)
    generate_n=$(echo "scale=10; $generate_n * $gen_train_ratio" | bc)
    generate_n=$(echo "$generate_n / 1" | awk '{print int($1) + ($1 > int($1))}')

    # Calculate the second term: ceil(250 / n_part)
    lower_bound=$(echo "scale=10; 250 / $n_part" | bc)
    lower_bound=$(echo "$lower_bound / 1" | awk '{print int($1) + ($1 > int($1))}')

    # Cap to lower bound
    if [ "$generate_n" -lt "$lower_bound" ]; then
        generate_n=$lower_bound
    fi
fi

echo "generate_n:"$generate_n

## mixed
#mixed_models_options=(
#  # mistral nemo
#  "/lustre/fsn1/projects/rech/imi/utu57ed/.cache/huggingface/hub/models--unsloth--Mistral-Nemo-Base-2407/snapshots/067a5371598bb01f5c7ce9c3c457e7f41f7da258"
#  # llama 3.1 8B
#  "/lustre/fsn1/projects/rech/imi/utu57ed/.cache/huggingface/hub/models--unsloth--Meta-Llama-3.1-8B/snapshots/3514c510ea4ba4d650522f467d4d0cef7de4a43c"
#  # qwen 2.5 7B
#  "/lustre/fsn1/projects/rech/imi/utu57ed/.cache/huggingface/hub/models--unsloth--Qwen2.5-7B/snapshots/ba9b4387ffef6e88435fee0dcdd97a637fb817fc"
#  # gemma 2 9B
#  "/lustre/fsn1/projects/rech/imi/utu57ed/.cache/huggingface/hub/models--unsloth--gemma-2-9b/snapshots/305ed9a1bf6aefcdba9fabdc0d4f9a6de413dcef"
#)

# mixed (small)
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

model_name=`echo $model | sed 's/.*unsloth--\([^\/]*\)\/snapshots.*/\1/'`
model_tag=${model_name//\//_}

source /linkhome/rech/genini01/utu57ed/.bashrc

module purge
module load arch/h100
module load python/3.11.5
conda activate unsloth_311

accumulate=1

dattype="standard"
#dattype="hq"
#dattype="ld"

epochs=1
#max_steps=-1
max_steps=4000
rank=16
alpha=16
per_device_batch_size=16
lr=2e-4
lr_scheduler="linear"
warmup_ratio=0.00125 # 5/4000; 5 steps

temp=1.5
min_p=0.2
n_generations=20
echo "n_generations:"$n_generations

exp_path=results/scale_small_4k_train_${model_tag}_dataset_${dataset_name}_type_${dattype}_presampled_split_${split}_acc_${accumulate}_ft_size_${per_participant_ft_dataset_size}_gen_train_ratio_${gen_train_ratio}_participants_${n_part}/generated_${per_participant_ai_dataset_size}_human_${per_participant_human_dataset_size}_unsloth/seed_${seed}_${datetime}

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
    --gen-train-dataset-size-ratio $gen_train_ratio \
    --human-dataset $dataset_name \
    --load-presampled-human-dataset \
    --split $split \
    --accumulate $accumulate \
    --dataset-type $dattype \
    --seed "${seed}_gen_${gen_i}" 2>&1 | tee -a $log_path

  for part_i in $(seq 0 $((n_part-1))); do
    echo "Part: "$part_i

    # Sample a random model if model == mixed
    if [[ "$model" == "mixed" ]]; then
      # Randomly select one model from the list
      selected_model=$(shuf -e "${mixed_models_options[@]}" -n 1)
    else
      selected_model=$model
    fi

    echo "Selected model "$selected_model

    save_dir=$exp_path"/gen_"$gen_i"/part_"$part_i

    python -u ft_and_gen.py \
        --save-dir $save_dir \
        --seed "${seed}_gen_${gen_i}_part_${part_i}" \
        --model-name $selected_model \
        --generate \
        --epochs $epochs \
        --max-steps $max_steps \
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
python evaluate_generations.py --emb --gib --seed-dir $exp_path

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
