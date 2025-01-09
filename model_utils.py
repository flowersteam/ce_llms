import warnings

import numpy as np
import time
import jsonlines
import torch
import random

from datasets import Dataset

from transformers import TrainingArguments, DataCollatorForSeq2Seq


try:
    from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
    from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig, PeftModel
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template, train_on_responses_only
except:
    warnings.warn("Packages for unsloth inference were not installed.")
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
        max_seq_length=max_model_len,
        dtype=torch.bfloat16,
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
    print(f"Model loading time: {loading_time} secs.")

    return model, tokenizer

instruction_template = "### User:\n"
response_template = "### Assistant:\n"
chat_template = instruction_template + "{}\n"+response_template+"{}"

def train_model_unsloth(tokenizer, model, dataset, save_dir, max_seq_length=256, train_args={}):

    # format dataset for model
    def formatting_func(examples):
        return {
            "text": [
                chat_template.format(ins, t) +
                tokenizer.eos_token
                for ins, t in zip(examples["instruction"], examples['text'])
            ]
        }

    dataset = dataset.map(formatting_func, batched=True, desc="Formatting training data", load_from_cache_file=False)

    collator = DataCollatorForSeq2Seq(tokenizer)

    training_arguments = TrainingArguments(output_dir=save_dir, **train_args)

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

    trainer = train_on_responses_only(
        trainer,
        instruction_part=instruction_template,
        response_part=response_template
    )

    # # verify that masking is done
    # print(tokenizer.decode(trainer.train_dataset[5]["input_ids"]))
    # star = tokenizer("*", add_special_tokens=False).input_ids[0]
    # tokenizer.decode([star if x == -100 else x for x in trainer.train_dataset[5]["labels"]])

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

    train_logs = {
        "model_save_path": str(final_state_dir),
        "training arguments": training_arguments.to_dict(),
        "train_result": train_result
    }

    return train_logs


def generate_data(
        all_instructions,
        model,
        generate_n_posts,
        tokenizer,
        batch_size=500,
        verbose=False,
        generation_arguments={},
        seed=None,
):
    s = time.time()
    # unsloth inference
    print("Patching model for inference")

    FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

    patching_time = time.time() - s
    print(f"Inference patching time: {patching_time}.")

    s = time.time()

    if len(all_instructions) < generate_n_posts:
        n_repeats = (generate_n_posts // len(all_instructions)) + 1
        all_instructions = n_repeats * all_instructions

    # sample instructions
    # shuffle
    instructions = random.sample(all_instructions, k=len(all_instructions))

    dataset_size = len(instructions)

    generated_texts = []
    used_instructions = []

    if generate_n_posts < batch_size:
        batch_size = generate_n_posts

    n_batches = int(np.ceil(generate_n_posts / batch_size))

    for batch_i in range(n_batches):
        if batch_i > 0:
            print(f"Generated posts {batch_i*batch_size}/{generate_n_posts}.")

            eta = (time.time() - s) / batch_i * (n_batches - batch_i)
            hours, minutes, seconds = secs_2_hms(eta)
            print("\tETA: %d:%02d:%02d" % (hours, minutes, seconds))

        # if last batch
        if len(generated_texts) + batch_size > generate_n_posts:
            batch_size = generate_n_posts - len(generated_texts)

        batch_start_i = (batch_i*batch_size) % dataset_size
        batch_end_i = (batch_start_i + batch_size) % dataset_size

        if batch_start_i < batch_end_i:
            # no wrap-around
            batch_indices = list(range(batch_start_i, batch_end_i))
        else:
            # wrap indices
            batch_indices = list(range(batch_start_i, dataset_size)) + list(range(0, batch_end_i))

        batch_instructions = [instructions[i] for i in batch_indices]
        input_ids = tokenizer(
            [chat_template.format(instr, "") for instr in batch_instructions],
            return_tensors="pt",
            padding=True
        ).to("cuda")['input_ids']

        batch_gen_ids = model.generate(
            input_ids=input_ids,
            **generation_arguments,
            use_cache=True,
        )

        for inp, gen_ids in zip(input_ids, batch_gen_ids):
            tx = tokenizer.decode(gen_ids[len(inp):], skip_special_tokens=True)
            generated_texts.append(tx)

        print(f"Generated text sample:{generated_texts[-1]}")

        used_instructions.extend(batch_instructions)

    output_dataset = Dataset.from_dict({
        "instruction": used_instructions,
        "text": generated_texts
    })

    generation_time = time.time() - s
    print(f"Generation time: {generation_time}.")

    # print a sample
    if verbose:
        for item in output_dataset.select(random.sample(range(dataset_size), 3)):
            p, tx = item['instruction'], item['text']
            print(f"\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n{p.strip()}\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n", tx)

    return output_dataset, {"generation_time": generation_time}

