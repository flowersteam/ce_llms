#!/bin/bash
#SBATCH -A imi@v100
#SBATCH -C v100-32g
#SBATCH --time=03:55:59
#SBATCH --gres=gpu:1
#SBATCH --array=0-4
#SBATCH -o logs/log_%A_%a.log
#SBATCH -e logs/log_%A_%a.log
##SBATCH --qos=qos_gpu-dev

cat $0

echo ""

#temp=0.7
temp=1.0
#temp=1.2

dataset_size=4000

model="unsloth/llama-3-8b-bnb-4bit"

#source ~/.bashrc
#module load python/3.10.4
#module load cuda/12.2.0
#module load cudnn/8.9.7.29-cuda
#conda activate unsloth_env

# conda activate cellm

datetime=`date +"%Y-%m-%d_%H-%M-%S"`

model_tag=${model//\//_}

n_part=2


seed=$SLURM_ARRAY_TASK_ID

#dataset_seed=1
dataset_seed=2

#exp_name=llama3_8b_load_n_${dataset_size}_gen_n_${dataset_size}_temp_${temp}_${model_tag}_date_${datetime}_dataset_seed_${dataset_seed}/$seed
exp_name=llama3_8b_load_n_${dataset_size}_gen_n_${dataset_size}_temp_${temp}_${model_tag}_dataset_seed_${dataset_seed}/seed_${seed}_date_${datetime}

exp_path=results/$exp_name
#exp_path=dev_results/$exp_name

exp_path=dev_results/llama3_8b_load_n_4000_gen_n_4000_temp_1.0_unsloth_llama-3-8b-bnb-4bit_dataset_seed_2/seed__date_2024-08-21_23-07-25


for gen_i in {0..10}
do

  echo "GEN: "$gen_i
  # prepare the training datasets
  if [ "$gen_i" -gt 0 ]; then
    echo "sample datasets"
    python sample_datasets.py --exp-path $exp_path --generation "$gen_i" --n-participants "$n_part" --seed "${seed}_gen_${gen_i}_part_${part_i}"
  fi


  for part_i in $(seq 0 $((n_part-1))); do
    echo "Part: "$part_i
    python ft_and_gen.py \
        --exp-path $exp_path \
        --model-name $model \
        --load-n $dataset_size \
        --gen-n $dataset_size \
        --temp $temp \
        --generation $gen_i \
        --participant-id $part_i \
        --deduplicate \
        --seed $seed"_gen_"$gen_i"_part_"$part_i \
        --dataset-seed $dataset_seed\

  done
done

#python evaluate_generations.py --experiment-dir results/$exp_name