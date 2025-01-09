# Installation

## Setup your conda environments

We need three environments
1) training with unsloth
2) evaluation with sklearn

### Unsloth conda env
Load the modules
```
module purge
module load python/3.11.5
```

Create the conda env 
```
conda create --name unsloth_311 \
    python=3.11 \
    pytorch-cuda=12.1 \
    pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers \
    -y
conda activate unsloth_311
```

Install packages
```
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes
pip install -r requirements_unsloth.txt
```


### eval conda env

```
module purge
module load python/3.11.5
```

```
conda create --name eval_311  python=3.11
conda activate eval_311
pip install -r requirements_eval.txt
```


Install vader_lexicon 
```commandline
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt_tab')"
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
# todo: update this with iterative train, run_on_node, and generate_plots scrips
python evaluate_generations.py --emb --experiment-dir results/Testing_iterative_learning_instructions_deduplicate_n_4000_temp_0.7 
python show_sample_generations.py --experiment-dir results/Testing_iterative_learning_instructions_deduplicate_n_4000_temp_0.7/ 
python visualize.py --metric div --directories eval_results/Testing_iterative_learning_*/part_*
```