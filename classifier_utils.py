import argparse
import os
import pickle
from turtle import pd
#import torch
import numpy as np
import torch
from tqdm import trange
from transformers import pipeline, BitsAndBytesConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig, AutoModel, AutoModelForCausalLM
#from together import Together



def load_classifier(model, neutral_option=False, logit_bias=False, use_together=False, multi_label=False):
    candidate_labels = None
    client = None
    if model == 'zero_shot':
        # zero shot
        political_nlp = pipeline(model="facebook/bart-large-mnli", task="zero-shot-classification")
        hypothesis_template = "The auhtor of this text is a {}."
        candidate_labels = ["Republican", "Democrat"]
        party_to_label = {"Republican": "Conservative", "Democrat": "Liberal"}
    elif model == 'DeBertA':
        # DeBertA
        label2id = {"Left": 0, "Center": 1, "Right": 2}
        id2label = {0: "Left", 1: "Center", 2: "Right"}

        lean_to_label = {"Left": "Liberal", "Center": "Neutral", "Right": "Conservative"}

        config = AutoConfig.from_pretrained("premsa/political-bias-prediction-allsides-DeBERTa", label2id=label2id, id2label=id2label)
        political_model = AutoModelForSequenceClassification.from_pretrained("premsa/political-bias-prediction-allsides-DeBERTa", config=config)
        political_tokenizer = AutoTokenizer.from_pretrained("premsa/political-bias-prediction-allsides-DeBERTa")
        
        political_nlp = pipeline("text-classification", model=political_model, tokenizer=political_tokenizer, top_k=None)
    elif model == 'PoliBERT':
        # choose GPU if available
       # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # select mode path here
        pretrained_LM_path = "kornosk/polibertweet-mlm"

        # load model
        political_tokenizer = AutoTokenizer.from_pretrained(pretrained_LM_path)
        political_nlp = AutoModel.from_pretrained(pretrained_LM_path)

        lean_to_label = {"democratic": "Liberal", "republican": "Conservative"}




    elif model == 'gpt-4o':
        system_prompt = """
                 You are a helpful assistant, expert in political lean classification. Your task is to classify the political lean of the text you receive. 
                 Output your answers as a label: 'L' for left-wing and 'R' for right-wing. Do not output any other information than the label.
                 """
        from openai import AzureOpenAI
        client = AzureOpenAI(
                            azure_endpoint=os.getenv(f"AZURE_OPENAI_ENDPOINT_gpt_4o_0513"),
                            api_key=os.getenv("AZURE_OPENAI_KEY_gpt_4o_0513"),
                            api_version=os.getenv("AZURE_OPENAI_API_VERSION_gpt_4o_0801"),
                        )
        

        

        lean_to_label = {"L": "Liberal", "R": "Conservative"}

    elif model == 'gpt-4o-mini': ## TODO
        system_prompt = """
                 You are a helpful assistant, expert in political lean classification. Your task is to classify the political lean of the text you receive. 
                 Output your answers as a label: 'L' for left-wing and 'R' for right-wing. Do not output any other information than the label.
                 """
        from openai import AzureOpenAI
        client = AzureOpenAI(
                            azure_endpoint=os.getenv(f"AZURE_OPENAI_ENDPOINT_gpt_4o_0513"),
                            api_key=os.getenv("AZURE_OPENAI_KEY_gpt_4o_0513"),
                            api_version=os.getenv("AZURE_OPENAI_API_VERSION_gpt_4o_0801"),
                        )
        

        

        lean_to_label = {"L": "Liberal", "R": "Conservative"}

    elif model == 'Llama-3.1-8B-Instruct' or model == 'Llama-3-70B-Instruct' or model == 'Llama-3.1-70B-Instruct' or  model == 'Llama-3.1-70B-Instruct-Turbo' or model == 'Mixtral-8x22B-Instruct-v0.1' or model == 'Mixtral-8x7B-Instruct-v0.1' or model == 'Mistral-7B-Instruct-v0.1' or model == 'Qwen2.5-72B-Instruct-Turbo':
        if 'Llama' in model:
            model_id = "meta-llama/Meta-" + model
        elif 'Mixtral' or 'Mistral' in model:
            model_id = "mistralai/" + model
        elif 'Qwen' in model:
            model_id = "Qwen/" + model


        if use_together:
            client = Together(api_key=os.getenv("TOGETHER_API_KEY"))

            system_prompt = """
                    You are a helpful assistant, expert in political lean classification. Your task is to classify the political lean of the text you receive. 
                    Output your answers as a label: 'L' for left-wing and 'R' for right-wing. Do not output any other information than the label.
                    """
            
        else: #use transformers

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            

            political_nlp = AutoModelForCausalLM.from_pretrained(model_id, device_map = "auto", trust_remote_code=True, torch_dtype = torch.bfloat16).eval()
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            tokenizer.pad_token_id = tokenizer.eos_token_id  # Set a padding token

            if neutral_option:
                system_prompt = """
                        You are a helpful assistant, expert in political lean classification. Your task is to classify the political lean of the text you receive. 
                        Output your answers as a label: 'L' for left-wing, 'R' for right-wing and 'N' for non-political. Do not output any other information than the label. Here is the text:
                        """

            else:

                system_prompt = """
                        You are a helpful assistant, expert in political lean classification. Your task is to classify the political lean of the text you receive. 
                        Output your answers as a label: 'L' for left-wing and 'R' for right-wing. Do not output any other information than the label. Here is the text:
                        """
                
                

            
        if neutral_option:
            lean_to_label = {"L": "Liberal", "R": "Conservative", "N": "Neutral"}    
        else:
            lean_to_label = {"L": "Liberal", "R": "Conservative"}

    else:
        print(f"Classifier {model} not recognized")
        return

    return political_nlp, system_prompt, lean_to_label, tokenizer, candidate_labels, client

def predict_political_lean(batch_texts, model, political_nlp, system_prompt, candidate_labels, tokenizer , client = None, neutral_option = True, use_together = False, logit_bias = False):
    
    
    
    
    if model == 'zero_shot':
                # batch_results = political_nlp(batch_texts, candidate_labels=candidate_labels, hypothesis_template=hypothesis_template)
                batch_results = political_nlp(batch_texts, candidate_labels=candidate_labels)
    elif model == 'DeBertA':
        batch_results = political_nlp(batch_texts)
    
    elif model == 'gpt-4o':
        results = []
        for text in batch_texts:
            message_text = [{"role": "system", "content": system_prompt}, {"role": "user", "content": text}]

            res = client.chat.completions.create(
                            model="gpt-4o", # model = "deployment_name"
                            messages = message_text,
                            temperature=0,
                            max_tokens=800,
                            top_p=0.95,
                            frequency_penalty=0,
                            presence_penalty=0,
                            stop=None
                            )
            results.append(res.choices[0].message.content)
        # print(batch_results)
        batch_results = [{"labels": [res]} for res in results]
    
    elif model == 'gpt-4o-mini':
        results = []
        for text in batch_texts:
            message_text = [{"role": "system", "content": system_prompt}, {"role": "user", "content": text}]

            res = client.chat.completions.create(
                            model="gpt-4o-mini-nobatch", # model = "deployment_name"
                            messages = message_text,
                            temperature=0,
                            max_tokens=800,
                            top_p=0.95,
                            frequency_penalty=0,
                            presence_penalty=0,
                            stop=None
                            )
            results.append(res.choices[0].message.content)
        # print(batch_results)
        batch_results = [{"labels": [res]} for res in results]

    elif model == 'Llama-3.1-8B-Instruct-Turbo' or model == 'Llama-3.1-70B-Instruct-Turbo' or "Qwen2.5-72B-Instruct-Turbo" or model == 'Mixtral-8x22B-Instruct-v0.1' or model == 'Mixtral-8x7B-Instruct-v0.1' or model == 'Mistral-7B-Instruct-v0.1':
        results = []
        scores = []
        lprobs = []
        probs = []
        leans = []
        if use_together:
            if model == 'Qwen2.5-72B-Instruct-Turbo':
                model_id="Qwen/Qwen2.5-72B-Instruct-Turbo"
            elif 'Llama' in model:
                model_id = "meta-llama/Meta-" + model
            elif 'Mixtral' or 'Mistral' in model:
                model_id = "mistralai/" + model
                

            for text in batch_texts:
            
                res = client.chat.completions.create(
                model=model_id,
                temperature=0,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": text}]
                )
                results.append(res.choices[0].message.content)
            
            batch_results = [{"labels": [res]} for res in results]


        else:
            conversations = [[{"role": "system", "content": system_prompt}, {"role": "user", "content": text}] for text in batch_texts]

            # texts = tokenizer.apply_chat_template([[system_prompt + text for text in batch_texts]])
            # output = llm.generate(inputs["input_ids"],return_dict_in_generate=True, max_new_tokens=1, sequence_bias=sequence_bias, output_scores=True)
            print("applying chat template")
            tokens = tokenizer.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)
            print('tokenizing')

            # inputs = tokenizer(tokens, return_tensors="pt")

            inputs = tokenizer(tokens, padding=True, return_tensors="pt").to(political_nlp.device)


            print(inputs)

            output = political_nlp.generate(
                **inputs,
                max_new_tokens=1,
                return_dict_in_generate=True,
                output_scores=True,
                # sequence_bias=sequence_bias,
                # use_cache=True,
            )
            if neutral_option:
                answers = ['L', 'R', 'N']
            else:
                answers = ['L', 'R']
            score, generation, lprob, prob, leans = parse_hf_outputs(output=output, answers=answers, tokenizer=tokenizer, neutral_option=neutral_option)
            #results.append(tokenizer.batch_decode(output, skip_special_tokens=True)[0])
            results.append(generation)
            scores.append(score)
            lprobs.append(lprob)
            probs.append(prob)
            leans.append(leans)


            batch_results = [{"labels": results, "scores": scores, "lprobs": lprobs, "probs": probs, "leans": leans}]

    return batch_results