import warnings

import numpy as np
import time
import jsonlines
import torch
import random

from datasets import Dataset

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments

try:
    from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
    from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig, PeftModel
    from unsloth import FastLanguageModel
except:
    warnings.warn("Packages for unsloth inference were not installed.")
    pass

try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
except:
    warnings.warn("Packages for vllm inference were not installed.")
    pass

def secs_2_hms(s):
    minutes, seconds = divmod(s, 60)
    hours, minutes = divmod(minutes, 60)
    return hours, minutes, seconds


def load_model(model_path, seed, r=16, alpha=16, use_rslora=False, load_in_4bit=True, max_model_len=2048):
    print(f"Loading the model from: {model_path}")
    s = time.time()

    assert ("4bit" in model_path) == load_in_4bit

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_model_len,  # Choose any! We auto support RoPE Scaling internally!
        dtype=None,  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
        load_in_4bit=load_in_4bit,
    )

    if hasattr(model.config, "model_max_length"):
        assert max_model_len <= model.config.model_max_length
    else:
        assert max_model_len <= model.config.max_position_embeddings

    print("Patching the model")
    model = FastLanguageModel.get_peft_model(
        model,
        r=r,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=alpha,
        lora_dropout=0,  # Supports any, but=0 is optimized
        bias="none",  # Supports any, but="none" is optimized
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=seed,
        use_rslora=use_rslora,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
    )

    loading_time = time.time() - s
    print(f"Model loading time: {loading_time}.")

    return model, tokenizer

response_template = "### Assistant:\n"
chat_template = "### User:\n{}\n"+response_template+"{}"

def train_model_unsloth(tokenizer, model, dataset, save_dir, save_merged=False, max_seq_length=256, train_args={}):

    # format dataset for model
    def formatting_func(examples):
        return {
            "text": [
                chat_template.format(ins, " "+t) +
                tokenizer.eos_token
                for ins, t in zip(examples["instruction"], examples['text'])
            ]
        }

    dataset = dataset.map(formatting_func, batched=True, desc="Formatting training data")

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
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    final_state_dir = save_dir / "final"
    trainer.model.save_pretrained(final_state_dir)
    trainer.tokenizer.save_pretrained(final_state_dir)
    print("Model adapters and tokenizer saved to:", final_state_dir)

    if save_merged:
        merge_s = time.time()
        full_model_save_dir = save_dir / "final_merged"
        trainer.model.save_pretrained_merged(full_model_save_dir, tokenizer, save_method="merged_16bit")
        print("Full model saved to :", full_model_save_dir)

        merging_and_saving_time = time.time() - merge_s
        hours, minutes, seconds = secs_2_hms(merging_and_saving_time)
        print("Model merging and saving time: %d:%02d:%02d" % (hours, minutes, seconds))

    train_logs = {
        "model_save_path": str(final_state_dir),
        "training arguments": training_arguments.to_dict(),
        "train_result": train_result
    }

    return train_logs


def generate_data(
        instructions,
        model,
        tokenizer=None,  # if vllm=False
        vllm=False,
        batch_size=500,
        verbose=False,
        generation_arguments={},
        seed=None,
):
    if not vllm:
        if tokenizer is None:
            raise ValueError("tokenizer must be provided if non-vllm inference is used")

    if not vllm:
        s = time.time()
        # unsloth inference
        print("Patching model for inference")
        FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

        patching_time = time.time() - s
        print(f"Inference patching time: {patching_time}.")

    s = time.time()

    print("Preparing prompts")
    # create a dataset for generation
    dataset = Dataset.from_dict({"instruction": instructions})
    dataset = dataset.map(lambda x: {"text": chat_template.format(x["instruction"], "")}, desc="Formatting generation instructions")
    dataset_size = len(dataset)

    generated_texts = []

    for b_i in range(0, dataset_size, batch_size):
        print(f"Generated {b_i}/{dataset_size}.")

        if b_i > 0:
            eta = (time.time() - s)/b_i * (dataset_size - b_i)
            hours, minutes, seconds = secs_2_hms(eta)
            print("\tETA: %d:%02d:%02d" % (hours, minutes, seconds))

        # batch_indices = np.random.randint(0, len(dataset), batch_size)
        batch_indices = list(range(b_i, min(b_i+batch_size, dataset_size)))
        batch = dataset.select(batch_indices)

        if vllm:
            outputs = model.generate(batch["text"], SamplingParams(**generation_arguments, seed=seed))
            generated_texts.extend([o.outputs[0].text for o in outputs])

        else:
            model_inputs = tokenizer(batch["text"], return_tensors="pt", padding=True).to("cuda")
            # todo: set seed?
            batch_gen_ids = model.generate(
                **model_inputs,
                **generation_arguments,
                use_cache=True,
            )

            for inp, gen_ids in zip(model_inputs['input_ids'], batch_gen_ids):
                tx = tokenizer.decode(gen_ids[len(inp):], skip_special_tokens=True)
                generated_texts.append(tx)

    output_dataset = Dataset.from_dict({
        "instruction": instructions,
        "text": generated_texts
    })

    generation_time = time.time() - s
    print(f"Generation time: {generation_time}.")

    # print a sample
    if verbose:
        for item in output_dataset.select(random.sample(range(dataset_size), 5)):
            p, tx = item['instruction'], item['text']
            print(f"\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n{p.strip()}\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n", tx)

    return output_dataset, {"generation_time": generation_time}

