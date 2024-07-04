#!/bin/bash
cat $0

#temp=0.7
temp=1.2
#temp=1.5
dataset_size=4000

for i in {0..6}
do
    python ft_and_gen.py \
        --exp-path results/Testing_iterative_learning_instructions_v2_deduplicate_load_n_${dataset_size}_gen_n_${dataset_size}_temp_${temp} \
        --load-n $dataset_size \
        --gen-n $dataset_size \
        --temp $temp \
        --generation $i \
        --deduplicate \

done