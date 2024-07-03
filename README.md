# Installation

conda create -n cellm python=3.10
conda activate cellms
pip install -r requirements.txt


# Experiment

bash iterative_train.sh

python evaluate_generations.py

python show_sample_generations.py --experiment-dir results/Testing_iterative_learning_instructions_deduplicate_n_4000_temp_0.7/ 


python visualize.py --metric cos_diversities eval_results/Testing_iterative_learning_*

