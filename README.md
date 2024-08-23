# Installation

## Setup your conda environment with unsloth

Load the modules
```
module load python/3.10.4
module load cuda/12.2.0
module load cudnn/8.9.7.29-cuda
```

Create the conda env 
```
conda create --name cellm python=3.10
conda activate cellm
```

Install unsloth
```
# you can use conda as well, but mamba is much faster
mamba install cudatoolkit "xformers<0.0.27" bitsandbytes pytorch pytorch-cuda=12.1 -c pytorch -c nvidia -c xformers -c conda-forge -y

pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" --no-cache-dir

pip install --no-cache-dir --no-deps "trl<0.9.0" "xformers<0.0.27" peft accelerate bitsandbytes
```

Install other requirements
```
pip install --no-cache-dir -r requirements_unsloth.txt
```

Install vader_lexicon 
```commandline
python -c "import nltk; nltk.download('vader_lexicon')"
```

## Setup your cache dir

Add the following line to `./bashrc`:
```commandline
export HF_HOME="<path_to_cache_dir>"
```

Reload `./bashrc` with `source ~/.bashrc`

## Download unsloth models

Install huggingface-cli
```commandline
conda install -c conda-forge huggingface_hub
```



Download a model to the hf_cache_dir
```
cd $HF_HOME
huggingface-cli download unsloth/llama-3-8b-bnb-4bit --local-dir unsloth/llama-3-8b-bnb-4bit --local-dir-use-symlinks True --cache-dir $HF_HOME
```



# Experiment

```
bash iterative_train.sh
```

```
python evaluate_generations.py --experiment-dir results/Testing_iterative_learning_instructions_deduplicate_n_4000_temp_0.7/ 
python show_sample_generations.py --experiment-dir results/Testing_iterative_learning_instructions_deduplicate_n_4000_temp_0.7/ 
python visualize.py --metric cos_diversities eval_results/Testing_iterative_learning_*
```