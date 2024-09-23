#!/bin/bash
#SBATCH -A imi@a100
#SBATCH -C a100
#SBATCH --time=01:55:59
#SBATCH --gres=gpu:1
#SBATCH --array=0-3
##SBATCH --array=0-0
#SBATCH -o logs/log_%A_%a.log
#SBATCH -e logs/log_%A_%a.log
#SBATCH --qos=qos_gpu-dev

cat $0

echo ""

temp=1.0

#per_participant_generated_dataset_size=4000
#per_participant_human_dataset_size=1000

# Assuming seed is set to the SLURM_ARRAY_TASK_ID environment variable
# todo: update this to 2D SLURM ARRAY TASK ID
# todo: IT SHOULD NOT BE LIKE THIS BECAUSE SEED=$SLURM_ARRAY_TASK_ID
ratio=$SLURM_ARRAY_TASK_ID

# human dataset size for generation 0
per_participant_ft_dataset_size=8000

# Use a case statement to set per_participant_generated_dataset_size and per_participant_human_dataset_size

case $ratio in
  0)
    per_participant_generated_dataset_size=8000
    per_participant_human_dataset_size=0
    ;;
  1)
    per_participant_generated_dataset_size=6000
    per_participant_human_dataset_size=2000
    ;;
  2)
    per_participant_generated_dataset_size=4000
    per_participant_human_dataset_size=4000
    ;;
  3)
    per_participant_generated_dataset_size=2000
    per_participant_human_dataset_size=6000
    ;;
  *)
    echo "Error: ratio value ($ratio) is not recognized."
    exit 1
    ;;
esac

model="unsloth/llama-3-8b-bnb-4bit"

#source ~/.bashrc
module load python/3.10.4
#module load cuda/12.2.0
#module load cudnn/8.9.7.29-cuda
conda activate llm_ce

datetime=`date +"%Y-%m-%d_%H-%M-%S"`

model_tag=${model//\//_}

n_part=2
#n_part=1


seed=$SLURM_ARRAY_TASK_ID

#dataset_seed=1
dataset_seed=2_${seed}

# seed
#exp_name=llama3_8b_load_n_${per_participant_generated_dataset_size}_gen_n_${per_participant_generated_dataset_size}_temp_${temp}_${model_tag}_dataset_seed_${dataset_seed}/seed_${seed}_date_${datetime}
#exp_path=results/$exp_name

#exp_path=dev_results/human_data_ratio_particiapnts_${n_part}_generated_dataset_size_${per_participant_generated_dataset_size}_human_dataset_size_${per_participant_human_dataset_size}/seed_${seed}_${datetime}

#exp_path=dev_results/human_data_ratio_particiapnts_${n_part}_generated_dataset_size_${per_participant_generated_dataset_size}_human_dataset_size_${per_participant_human_dataset_size}/seed_${seed}_${datetime}

exp_path=dev_results/human_data_ratio_particiapnts_${n_part}_generated_dataset_size_${per_participant_generated_dataset_size}_human_dataset_size_${per_participant_human_dataset_size}/seed_${seed}_${datetime}

for gen_i in {0..9}
do

  echo "GEN: "$gen_i
  # prepare the training datasets
  if [ "$gen_i" -eq 0 ]; then
    # in first the first generation
    current_per_participant_human_dataset_size=$per_participant_ft_dataset_size

  else
    current_per_participant_human_dataset_size=$per_participant_human_dataset_size

  fi

  echo $current_per_participant_human_dataset_size

  python sample_datasets.py \
    --exp-path $exp_path --generation "$gen_i" --n-participants "$n_part" \
    --per-participant-human-dataset-size $current_per_participant_human_dataset_size \
    --human-dataset "twitter" \
    --human-dataset-seed "${seed}_gen_${gen_i}" \
    --seed "${seed}_gen_${gen_i}"


  for part_i in $(seq 0 $((n_part-1))); do
    echo "Part: "$part_i
    python ft_and_gen.py \
        --exp-path $exp_path \
        --model-name $model \
        --gen-n $per_participant_generated_dataset_size \
        --temp $temp \
        --generation $gen_i \
        --participant-id $part_i \
        --deduplicate \
        --seed $seed"_gen_"$gen_i"_part_"$part_i \
#        --dataset-seed "${dataset_seed}_part_${part_i}"\

  done
done

# python evaluate_generations.py --experiment-dir results/$exp_name