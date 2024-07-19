import numpy as np
import time
import jsonlines
import torch
import random

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,HfArgumentParser,TrainingArguments, pipeline, logging, DataCollatorForLanguageModeling, DataCollatorWithPadding
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig, PeftModel

from unsloth import FastLanguageModel

def secs_2_hms(s):
    minutes, seconds = divmod(s, 60)
    hours, minutes = divmod(minutes, 60)
    return hours, minutes, seconds


def load_finetuned_model_with_adapters(model_path, bnb_config=None, cache_dir=None):
    print("Loading from : ", model_path)

    lora_config = LoraConfig.from_pretrained(model_path)

    base_model = lora_config.base_model_name_or_path

    if bnb_config is None:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False,
        )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=cache_dir
    )
    model = prepare_model_for_kbit_training(model)
    model = PeftModel.from_pretrained(model, model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        lora_config.base_model_name_or_path, trust_remote_code=True, cache_dir=cache_dir)
    return model, tokenizer


def load_model(model_name, cache_dir):
    # assert model_name == "mistralai/Mistral-7B-v0.1"

    # LOAD TOKENIZER
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir)
    tokenizer.padding_side = 'right'
    # tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token = tokenizer.unk_token  # no grads on pad_token
    print("Pad with unk token.")
    tokenizer.add_eos_token = True

    # LOAD MODEL
    # bnb_config=None
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=cache_dir,
        # attn_implementation="flash_attention_2",
    )

    return tokenizer, model


def prepare_model_for_training(tokenizer, model, peft_config):
    # prepare for training
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    model.config.pretraining_tp = 1
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    # add adapters
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return tokenizer, model


def load_model(model_path, seed):
    print(f"Loading the model from: {model_path}")

    max_seq_length = 8192  # LLama3

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,  # Choose any! We auto support RoPE Scaling internally!
        dtype=None,  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
        load_in_4bit=True,
    )

    if hasattr(model.config, "model_max_length"):
        assert max_seq_length <= model.config.model_max_length
    else:
        assert max_seq_length <= model.config.max_position_embeddings

    print("Patching the model")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,  # Supports any, but=0 is optimized
        bias="none",  # Supports any, but="none" is optimized
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=seed,
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
    )

    return model, tokenizer

response_template = "### Assistant:\n"
chat_template = "### User:\n{}\n"+response_template+"{}"

def train_model_unsloth(tokenizer, model, dataset, save_dir, instructions, max_seq_length=256, train_args={}):

    # format dataset for model
    def formatting_func(examples):
        return {
            "text": [
                chat_template.format(random.choice(instructions), " "+t) +
                tokenizer.eos_token
                for t in examples['text']
            ]
        }

    dataset = dataset.map(formatting_func, batched=True)

    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False
    )

    # check that collator finds the response_template
    assert collator([tokenizer(dataset['text'][0])])['labels'][0].unique().numel() > 1

    training_arguments = TrainingArguments(
        output_dir=save_dir, **train_args
    )

    print("Training")
    # Setting sft parameters
    trainer = SFTTrainer(
        model=model,
        args=training_arguments,
        train_dataset=dataset,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        dataset_text_field="text",
        data_collator=collator,
        packing=False,
        dataset_num_proc=2,
    )

    train_result = trainer.train()

    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    print(f"{train_result.metrics['train_runtime']} seconds used for training.")
    print(f"{round(train_result.metrics['train_runtime'] / 60, 2)} minutes used for training.")
    print(f"Peak reserved memory = {used_memory} GB.")

    print("model was trained")
    # save the model
    metrics = train_result.metrics
    max_train_samples = train_result.global_step if train_result.global_step is not None else len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    final_state_dir = save_dir / "final"
    trainer.model.save_pretrained(final_state_dir)
    print("Model saved to:", final_state_dir)

    train_logs = {
        "model_save_path": str(final_state_dir),
        "training arguments": training_arguments.to_dict(),
        "train_result": train_result
    }

    return train_logs


def generate_data(
        tokenizer,
        model,
        instructions,
        save_dir=None,
        n_posts_to_generate=1000,
        batch_size=250,
        verbose=False,
        deduplicate=False,
        generation_arguments={},
):
    FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

    s = time.time()

    # Prepare inputs (instructions)
    print("Preparing prompts")
    prompts = [random.choice(instructions) for _ in range(np.minimum(batch_size, n_posts_to_generate))]
    dataset = Dataset.from_dict({"instruction": instructions})
    dataset = dataset.map(lambda x: {
        "text": chat_template.format(x["instruction"], "")
    })

    generated_texts = []

    for b_i in range(0, n_posts_to_generate, batch_size):
        print(f"Generated {b_i}/{n_posts_to_generate}.")

        if b_i > 0:
            eta = (time.time() - s)/b_i * (n_posts_to_generate - b_i)
            hours, minutes, seconds = secs_2_hms(eta)
            print("\tETA: %d:%02d:%02d" % (hours, minutes, seconds))

        batch_indices = np.random.randint(0, len(dataset), batch_size)
        batch = dataset.select(batch_indices)

        model_inputs = tokenizer(
            batch["text"],
            return_tensors="pt", padding=True
        ).to("cuda")

        # Generate posts
        batch_gen_ids = model.generate(
            **model_inputs,
            **{
                **generation_arguments,
                **{"use_cache": True}
            }
        )

        for inp, gen_ids in zip(model_inputs['input_ids'], batch_gen_ids):
            tx = tokenizer.decode(gen_ids[len(inp):], skip_special_tokens=True)
            generated_texts.append(tx)

    if deduplicate:
        generated_texts = list(set(generated_texts))
        print(f"N posts after deduplication: {len(generated_texts)}")

    generation_time = time.time() - s
    print(f"Generation time: {generation_time}.")

    # print a sample
    if verbose:
        for i in random.sample(range(n_posts_to_generate), 5):
            p, tx = prompts[i], generated_texts[i]
            print(f"\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n{p.strip()}\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n", tx)

    # save
    generated_text_path = None
    if save_dir:
        generated_text_path = save_dir / 'output.jsonl'
        with jsonlines.open(generated_text_path, "w") as writer:
            writer.write_all(zip(prompts, generated_texts))
            print(f"Generated text saved in {generated_text_path}.")

    return generated_texts, {"generation_time": generation_time, "generation_save_path": str(generated_text_path)}

