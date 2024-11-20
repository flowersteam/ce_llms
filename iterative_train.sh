#!/bin/bash
##SBATCH -A imi@a100
##SBATCH -C a100
#SBATCH -A vgw@h100
#SBATCH --cpus-per-task=24
#SBATCH -C h100
#SBATCH --time=09:55:59
#SBATCH --gres=gpu:1
#SBATCH --array=0-17
#SBATCH -o logs/iterative_train_%A_%a.log
#SBATCH -e logs/iterative_train_%A_%a.log
#SBATCH -J iterative_train
##SBATCH --qos=qos_gpu_h100-dev

start_time=$(date +%s)

cat $0

echo ""

echo "SLURM_JOB_ID: "$SLURM_JOB_ID"_"$SLURM_ARRAY_TASK_ID | tee -a $log_path

# 6 ratios
ratio_id=$((SLURM_ARRAY_TASK_ID % 6))
seed_id=$((SLURM_ARRAY_TASK_ID / 6))

datetime=`date +"%Y-%m-%d_%H-%M-%S"`
datetime_nano=`date +"%Y-%m-%d_%H-%M-%S.%N"`
seed=${seed_id}_${datetime_nano}

echo "ratio_id:"$ratio_id
echo "seed:"$seed

# human dataset size for generation 0
#per_participant_ft_dataset_size=8000
per_participant_ft_dataset_size=4000

# Define ratio array to use for calculating the generated dataset size
ratios=(0.0625 0.125 0.25 0.5 0.75 1)
per_participant_ai_dataset_size=$(echo "${ratios[$ratio_id]} * $per_participant_ft_dataset_size / 1" | bc)
per_participant_human_dataset_size=$(( per_participant_ft_dataset_size - per_participant_ai_dataset_size ))

echo "ft_size:"$per_participant_ft_dataset_size
echo "generated_dataset_size:"$per_participant_ai_dataset_size
echo "human_dataset_size:"$per_participant_human_dataset_size

# mistral nemo
#model="/lustre/fsn1/projects/rech/imi/utu57ed/.cache/huggingface/hub/models--unsloth--Mistral-Nemo-Base-2407/snapshots/067a5371598bb01f5c7ce9c3c457e7f41f7da258"
#quantization="False"

# llama
model="/lustre/fsn1/projects/rech/imi/utu57ed/.cache/huggingface/hub/models--unsloth--Meta-Llama-3.1-8B/snapshots/3514c510ea4ba4d650522f467d4d0cef7de4a43c/"
quantization="False"

# qwen
#model="/lustre/fsn1/projects/rech/imi/utu57ed/.cache/huggingface/hub/models--unsloth--Qwen2.5-7B/snapshots/ba9b4387ffef6e88435fee0dcdd97a637fb817fc"
#quantization="False"

# gemma
#model="/lustre/fsn1/projects/rech/imi/utu57ed/.cache/huggingface/hub/models--unsloth--gemma-2-9b/snapshots/305ed9a1bf6aefcdba9fabdc0d4f9a6de413dcef"
#quantization="False"

#dataset_name="twitter"
dataset_name="webis_reddit"

model_name=`echo $model | sed 's/.*unsloth--\([^\/]*\)\/snapshots.*/\1/'`
model_tag=${model_name//\//_}

source /linkhome/rech/genini01/utu57ed/.bashrc

module purge
module load arch/h100
module load python/3.11.5
conda activate unsloth_311

#n_part=4
#n_part=2
n_part=1

#split="train"
split="test"


roof_prob=0.03

epochs=1
#epochs=0.001
rank=16
alpha=16
rslora="False"
per_device_batch_size=16
lr=2e-4
lr_scheduler="linear"
warmup_ratio=0.00125 # 5/4000; 5 steps

temp=1.5
min_p=0.2
roof_prob=0.03

accumulate=1

exp_path=dev_results/human_ai_ratio_v3_acc_${accumulate}_epochs_${epochs}_split_${split}_rank_${rank}_alpha_${alpha}_rslora_${rslora}_bs_${per_device_batch_size}_lr_${lr}_lr_sched_${lr_scheduler}_warmup_ratio_${warmup_ratio}_temp_${temp}_min_p_${min_p}_webis_reddit_ft_size_${per_participant_ft_dataset_size}_${model_tag}_participants_${n_part}_roof_prob_${roof_prob}/generated_${per_participant_ai_dataset_size}_human_${per_participant_human_dataset_size}_unsloth/seed_${seed}_${datetime}

mkdir -p $exp_path
log_path=$exp_path/log.txt

for gen_i in {0..19}
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
    --roof-prob $roof_prob \
    --split $split \
    --deduplicate \
    --accumulate $accumulate \
    --seed "${seed}_gen_${gen_i}" 2>&1 | tee -a $log_path


  for part_i in $(seq 0 $((n_part-1))); do
    echo "Part: "$part_i

    save_dir=$exp_path"/gen_"$gen_i"/part_"$part_i

    python -u ft_and_gen.py \
        --save-dir $save_dir \
        --seed "${seed}_gen_${gen_i}_part_${part_i}" \
        --model-name $model \
        --generate \
        --epochs $epochs \
        --rank $rank \
        --alpha $alpha \
        --use-rslora $rslora \
        --per-device-batch-size $per_device_batch_size \
        --lr $lr \
        --lr-scheduler $lr_scheduler \
        --warmup-ratio $warmup_ratio \
        --quantization $quantization \
        --roof-prob $roof_prob \
        --temp $temp \
        --min-p $min_p \
        --gen-unique-n $per_participant_ai_dataset_size \
        --dataset-name $dataset_name \
        --split $split 2>&1 | tee -a $log_path

  done
done


conda activate eval_311

for part_i in $(seq 0 $((n_part-1))); do
  python evaluate_generations.py --emb --tox --experiment-dir $exp_path --participant part_${part_i}
done

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
