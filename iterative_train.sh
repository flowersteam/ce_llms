#!/bin/bash
#SBATCH -A imi@v100
#SBATCH -C v100-32g
#SBATCH --time=03:55:59
#SBATCH --gres=gpu:1
#SBATCH -o logs/log_%A_%a.log
#SBATCH -e logs/log_%A_%a.log
##SBATCH --qos=qos_gpu-dev

cat $0

#temp=0.7
temp=1.0
#temp=1.2

dataset_size=4000

model="unsloth/llama-3-8b-bnb-4bit"

source ~/.bashrc
module load python/3.10.4
module load cuda/12.2.0
module load cudnn/8.9.7.29-cuda
conda activate unsloth_env

# conda activate cellm

datetime=`date +"%Y-%m-%d_%H-%M-%S"`

model_tag=${model//\//_}

exp_name=unsloth_llama3_8b_no_urls_load_n_${dataset_size}_gen_n_${dataset_size}_temp_${temp}_${model_tag}_date_${datetime}
#exp_name=TEST_${datetime}

for i in {0..10}
do
    python ft_and_gen.py \
        --exp-path results/$exp_name \
        --model-name $model \
        --load-n $dataset_size \
        --gen-n $dataset_size \
        --temp $temp \
        --generation $i \
        --deduplicate \
        --unsloth \
        --seed 1 \
        --dataset-seed 1 \

done

python evaluate_generations.py --experiment-dir results/$exp_name