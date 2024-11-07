#!/bin/bash
#SBATCH -A imi@a100
#SBATCH -C a100
#SBATCH --time=05:55:59
#SBATCH --gres=gpu:1
#SBATCH --array=0-3
#SBATCH -o logs/iterative_train_%A_%a.log
#SBATCH -e logs/iterative_train_%A_%a.log
#SBATCH -J iterative_train
##SBATCH --qos=qos_gpu_a100-dev

start_time=$(date +%s)

cat $0

echo ""


# grid search generatio params
#min_p= [0.1 0.05]
#t=[1.5 2]

# 1
#temp=1.0
#min_p=0.1

# 2
#temp=1.0
#min_p=0.05

# 3
#temp=1.5
#min_p=0.1

## 4
#temp=1.5
#min_p=0.05

# 5
#temp=2.0
#min_p=0.1

# 6
#temp=2.0
#min_p=0.05

# 7
temp=1.5
min_p=0.2

# 8
#temp=2.0
#min_p=0.1



#per_participant_generated_dataset_size=4000
#per_participant_human_dataset_size=1000

# Assuming seed is set to the SLURM_ARRAY_TASK_ID environment variable
# todo: update this to 2D SLURM ARRAY TASK ID
# todo: IT SHOULD NOT BE LIKE THIS BECAUSE SEED=$SLURM_ARRAY_TASK_ID
ratio=$SLURM_ARRAY_TASK_ID

# human dataset size for generation 0
#per_participant_ft_dataset_size=8000
per_participant_ft_dataset_size=4000

# Use a case statement to set per_participant_generated_dataset_size and per_participant_human_dataset_size
case $ratio in
  0)
#    per_participant_generated_dataset_size=500
    per_participant_generated_dataset_size=250
    ;;
  1)
#    per_participant_generated_dataset_size=1000
    per_participant_generated_dataset_size=500
    ;;
  2)
#    per_participant_generated_dataset_size=2000
    per_participant_generated_dataset_size=1000
    ;;
  3)
#    per_participant_generated_dataset_size=4000
    per_participant_generated_dataset_size=2000
    ;;
  4)
#    per_participant_generated_dataset_size=6000
    per_participant_generated_dataset_size=3000
    ;;
  5)
#    per_participant_generated_dataset_size=8000
    per_participant_generated_dataset_size=4000
    ;;
  *)
    echo "Error: ratio generated/humam ($ratio) is not recognized."
    exit 1
    ;;
esac

per_participant_human_dataset_size=$(( per_participant_ft_dataset_size - per_participant_generated_dataset_size ))


# mistral nemo
#model="/lustre/fsn1/projects/rech/imi/utu57ed/.cache/huggingface/hub/models--unsloth--Mistral-Nemo-Base-2407-bnb-4bit/snapshots/20cfd0e98fb2628b00867147b2c6f423d27f3561/"

# llama-4bit
#model="/lustre/fsn1/projects/rech/imi/utu57ed/.cache/huggingface/hub/models--unsloth--Meta-Llama-3.1-8B-bnb-4bit/snapshots/a8b0fc584b10e0110e04f9d21c7f10d24391c1d5/"

# llama-16bit
model="/lustre/fsn1/projects/rech/imi/utu57ed/.cache/huggingface/hub/models--unsloth--Meta-Llama-3.1-8B/snapshots/3514c510ea4ba4d650522f467d4d0cef7de4a43c/"


# qwen
#model="/lustre/fsn1/projects/rech/imi/utu57ed/.cache/huggingface/hub/models--unsloth--Qwen2.5-7B-bnb-4bit/snapshots/9d7326d436359f5a2033cc7a5c7daad8bbbb4bae/"

# gemma
#model="/lustre/fsn1/projects/rech/imi/utu57ed/.cache/huggingface/hub/models--unsloth--gemma-2-9b-bnb-4bit/snapshots/4a92d80aa6fecc1f41eb488431d835747e23ec75/"

#dataset_name="twitter"
dataset_name="webis_reddit"

model_name=`echo $model | sed 's/.*unsloth--\([^\/]*\)\/snapshots.*/\1/'`
model_tag=${model_name//\//_}

source /linkhome/rech/genini01/utu57ed/.bashrc


datetime=`date +"%Y-%m-%d_%H-%M-%S"`
datetime_nano=`date +"%Y-%m-%d_%H-%M-%S.%N"`

#n_part=2
n_part=1

seed=${SLURM_ARRAY_TASK_ID}_${datetime_nano}

roof_prob=0.03

epochs=1
rank=16
alpha=32

# seed
exp_path=dev_results/more_ratios_rank_${rank}_alpha_${alpha}_temp_${temp}_min_p_${min_p}_human_ai_ratio_webis_reddit_ft_size_${per_participant_ft_dataset_size}_${model_tag}_particiapnts_${n_part}_roof_prob_${roof_prob}/generated_${per_participant_generated_dataset_size}_human_${per_participant_human_dataset_size}_unsloth_vllm/seed_${seed}_${datetime}
#exp_path=dev_results/test_/seed_${seed}_${datetime}

mkdir -p $exp_path
log_path=$exp_path/log.txt

echo "SLURM_JOB_ID: "$SLURM_JOB_ID"_"$SLURM_ARRAY_TASK_ID | tee -a $log_path


for gen_i in {0..19}
do
  module purge
  module load python/3.11.5
  module load arch/h100
  conda activate ce_llm_311

  echo -e "\033[32mGEN: $gen_i\033[0m"
  # prepare the training datasets
  if [ "$gen_i" -eq 0 ]; then
    # in first the first generation
    current_per_participant_human_dataset_size=$per_participant_ft_dataset_size

  else
    current_per_participant_human_dataset_size=$per_participant_human_dataset_size

  fi

  python sample_datasets.py \
    --exp-path $exp_path --generation "$gen_i" --n-participants "$n_part" \
    --per-participant-human-dataset-size $current_per_participant_human_dataset_size \
    --human-dataset $dataset_name \
    --roof-prob $roof_prob \
    --seed "${seed}_gen_${gen_i}" 2>&1 | tee -a $log_path

  for part_i in $(seq 0 $((n_part-1))); do
    echo "Part: "$part_i

    save_dir=$exp_path"/gen_"$gen_i"/part_"$part_i

    python ft_and_gen.py \
        --save-dir $save_dir \
        --seed $seed"_gen_"$gen_i"_part_"$part_i \
        --model-name $model \
        --epochs $epochs \
        --rank $rank \
        --alpha $alpha \
        --roof-prob $roof_prob \
        --save-merged 2>&1 | tee -a $log_path


    module purge
    module load python/3.10.4
    conda activate vllm

    merged_model_dir=$save_dir"/model/final_merged"

    python gen.py \
        --model-name $merged_model_dir \
        --save-dir $save_dir \
        --seed "${seed}_gen_${gen_i}_part_${part_i}" \
        --dataset-name $dataset_name \
        --vllm \
        --roof-prob $roof_prob \
        --temp $temp \
        --min-p $min_p \
        --gen-n $per_participant_generated_dataset_size \
        --dataset-name $dataset_name \
        --deduplicate 2>&1 | tee -a $log_path

    # delete the full model save
    rm -rf $merged_model_dir

  done
done


conda activate eval_311
python evaluate_generations.py --emb --tox --experiment-dir $exp_path

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
