#!/bin/bash
#SBATCH -A imi@a100
#SBATCH -C a100
#SBATCH --time=01:55:59
#SBATCH --gres=gpu:1
#SBATCH --array=1-3
##SBATCH --array=0-0
#SBATCH -o logs/log_%A_%a.log
#SBATCH -e logs/log_%A_%a.log
#SBATCH --qos=qos_gpu-dev

cat $0

echo ""

temp=1.0

#generated_dataset_size=4000
#human_dataset_size=1000

# Assuming seed is set to the SLURM_ARRAY_TASK_ID environment variable
# todo: update this to 2D SLURM ARRAY TASK ID
# todo: IT SHOULD NOT BE LIKE THIS BECAUSE SEED=$SLURM_ARRAY_TASK_ID
ratio=$SLURM_ARRAY_TASK_ID

# Use a case statement to set generated_dataset_size and human_dataset_size
case $ratio in
  0)
    generated_dataset_size=4000
    human_dataset_size=0
    ;;
  1)
    generated_dataset_size=3000
    human_dataset_size=1000
    ;;
  2)
    generated_dataset_size=2000
    human_dataset_size=2000
    ;;
  3)
    generated_dataset_size=1000
    human_dataset_size=3000
    ;;
  *)
    echo "Error: ratio value ($ratio) is not recognized."
    exit 1
    ;;
esac

model="unsloth/llama-3-8b-bnb-4bit"

#source ~/.bashrc
module load python/3.10.4
module load cuda/12.2.0
module load cudnn/8.9.7.29-cuda
conda activate cellm_v2

datetime=`date +"%Y-%m-%d_%H-%M-%S"`

model_tag=${model//\//_}

n_part=2
#n_part=1


seed=$SLURM_ARRAY_TASK_ID

#dataset_seed=1
dataset_seed=2_${seed}

# seed
#exp_name=llama3_8b_load_n_${generated_dataset_size}_gen_n_${generated_dataset_size}_temp_${temp}_${model_tag}_dataset_seed_${dataset_seed}/seed_${seed}_date_${datetime}

#exp_path=results/$exp_name

exp_path=dev_results/human_data_ratio_particiapnts_${n_part}_generated_dataset_size_${generated_dataset_size}_human_dataset_size_${human_dataset_size}/seed_${seed}_${datetime}

#exp_path=dev_results/test

for gen_i in {0..9}
do

  echo "GEN: "$gen_i
  # prepare the training datasets
  if [ "$gen_i" -gt 0 ]; then
    echo "sample datasets"
    python sample_datasets.py \
      --exp-path $exp_path --generation "$gen_i" --n-participants "$n_part" \
      --human-dataset-size $human_dataset_size \
      --human-dataset "twitter" \
      --human-dataset-seed "${seed}_gen_${gen_i}_part_${part_i}" \
      --seed "${seed}_gen_${gen_i}_part_${part_i}"

#      --human-dataset-lean "Liberal" \ # todo: lean per participant
#      --human-dataset-lean "Conservative" \ # todo: lean per participant

  fi

  # todo: move human dataset gen-0 to sample_dataset.py -> ft_and_gen just loads a dataset
  for part_i in $(seq 0 $((n_part-1))); do
    echo "Part: "$part_i
    python ft_and_gen.py \
        --exp-path $exp_path \
        --model-name $model \
        --load-n $generated_dataset_size \
        --gen-n $generated_dataset_size \
        --temp $temp \
        --generation $gen_i \
        --participant-id $part_i \
        --deduplicate \
        --seed $seed"_gen_"$gen_i"_part_"$part_i \
        --dataset-seed "${dataset_seed}_part_${part_i}"\

  done
done

# python evaluate_generations.py --experiment-dir results/$exp_name