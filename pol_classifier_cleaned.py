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

import transformers
transformers.logging.set_verbosity_error()


torch.backends.cuda.enable_cudnn_sdp(False)

def parse_hf_outputs(output, answers, tokenizer, neutral_option=False):
        
        def extract_answer_tokens(answers):
            answer_tokens = {ans: tokenizer.encode(ans, add_special_tokens=False) for ans in answers}
            return answer_tokens

        # answer_tokens = extract_answer_tokens(answers)  # todo: repetitive -> extract

        generation = []
        probs = []
        leans = []
        option_scores = []
        lprobs = []


        for choice in output.choices:
            # option_score = {
            #     ans: max([output.scores[0][i, ind] for ind in answer_tokens[ans]])
            #     for ans in answers
            # }

            # take the most probable answer as the generation
            gen = choice.message.content

            logprobs_content = choice.logprobs.content[0].top_logprobs

            # Extract log probabilities for the tokens in `answers`
            lprob_dict = {}
            for entry in logprobs_content:
                token = entry.token
                if token in answers:
                    lprob_dict[token] = entry.logprob
            

            lprob = [lprob_dict[ans] if ans in lprob_dict else -np.inf for ans in answers]


            if neutral_option and (lprob[0] == np.inf and lprob[1] == np.inf and lprob[2] == np.inf) or (lprob[0] == -np.inf and lprob[1] == -np.inf and lprob[2] == -np.inf):
                    prob = [1/3, 1/3, 1/3]
            elif  not(neutral_option) and ((lprob[0] == np.inf and lprob[1] == np.inf) or (lprob[0] == -np.inf and lprob[1] == -np.inf)):
                prob = [0.5, 0.5]
        
            else:

                

                # use softmax to get probabilities
                prob = [np.exp(lp) / sum([np.exp(lp) for lp in lprob]) for lp in lprob]

            # todo: check that ' A' are one token and check for those as well and not "unk"
            # encoded_ans = [tokenizer.encode(ans, add_special_tokens=False)[0] for ans in answers]
            # option_score = {enc_a: output.scores[0][0, enc_a] for enc_a in encoded_ans}

            if gen == 'N': ## If text if neutral, lean is 0
                lean = 0
            else: #Otherwise, lean is Right - Left 
                lean = prob[1] - prob[0]


            

            generation.append(gen)
            probs.append(prob)
            leans.append(lean)
            option_scores.append(option_scores)
        



        return option_scores, generation, lprobs, probs, leans



## Function to evaluate accuracy

import random
from sklearn.utils import shuffle  # You can use this if you prefer sklearn's shuffle


def classifier_agreement(dataset, n_samples=1000, batch_size=1):
    ## This fucntion will take a list of texts and return the texts on which the two classifiers agree and are correct
    if dataset == 'all':
        dataset_names = ["twitter", "reddit", "reddit2", "news"]
    else:
        dataset_names = [dataset]
    texts_to_keep = {d: [] for d in dataset_names}
    for dataset_name in dataset_names:
        data = load_human_dataset(dataset_name)
        print(f"Loaded {len(data)} samples from {dataset_name}")
        
        # Shuffle the data to ensure random sampling
        # random.shuffle(data)  # You can also use sklearn's shuffle(data) if needed
        
        correct_zero_shot = 0
        correct_DeBertA = 0
        correct_agreement = 0
        incorrect_agreement = 0
        agreement = 0

        samples = 0
        
        texts = [row['text'] for row in data]
        labels = [row['Political Lean'] for row in data]

        # Shuffle the data to ensure random sampling, while keeping mapping between texts and labels
        texts, labels = shuffle(texts, labels, random_state=42)  


        zero_shot_nlp =  pipeline(model="facebook/bart-large-mnli", task="zero-shot-classification")
        hypothesis_template = "The political lean of this text is {}."
        candidate_labels = ["Conservative", "Liberal"]


        label2id = {"Left": 0, "Center": 1, "Right": 2}
        id2label = {0: "Left", 1: "Center", 2: "Right"}
        deBertA_config = AutoConfig.from_pretrained("premsa/political-bias-prediction-allsides-DeBERTa", label2id=label2id, id2label=id2label)
        deBertA_model = AutoModelForSequenceClassification.from_pretrained("premsa/political-bias-prediction-allsides-DeBERTa", config=deBertA_config)
        deBertA_tokenizer = AutoTokenizer.from_pretrained("premsa/political-bias-prediction-allsides-DeBERTa")
        deBertA_nlp = pipeline("text-classification", model=deBertA_model, tokenizer=deBertA_tokenizer)
        lean_to_label = {"Left": "Liberal", "Center": "Neutral", "Right": "Conservative"}

        
        for i in trange(0, len(data), batch_size):
            # print(f"{correct_agreement} / {n_samples}" )
            if correct_agreement >= n_samples:
                break
            #prepare batch
            text_batch = texts[i:i+batch_size]
            label_batch = labels[i:i+batch_size]

            
            
          
            zero_shot_res = zero_shot_nlp(text_batch, candidate_labels=candidate_labels, hypothesis_template=hypothesis_template)[0]
            deBertA_res = deBertA_nlp(text_batch)[0]
            
            #iterate through the batch
            for j in range(len(text_batch)):
                samples += 1
                text = text_batch[j]
                label = label_batch[j]
                zero_shot_label = zero_shot_res['labels'][0]
                deBertA_label = lean_to_label[deBertA_res['label']]

                # print(zero_shot_label, deBertA_label, label)
                if zero_shot_label == deBertA_label:
                    agreement += 1
                    if zero_shot_label == label:
                        correct_agreement += 1
                        samples += 1
                        texts_to_keep[dataset_name].append(text)
                    else:
                        incorrect_agreement += 1

        
        print(f"Dataset : {dataset_name}\nAgreement: {agreement/samples}\n Correct agreement: {correct_agreement/agreement}\n Incorrect agreement: {incorrect_agreement/agreement}")
        ##save the results to a text file
        with open(f"agreement_results_{dataset_name}_{n_samples}samples.txt", "w") as f:
            f.write(f"Dataset : {dataset_name}\nAgreement: {agreement/samples}\n Correct agreement: {correct_agreement/agreement}\n Incorrect agreement: {incorrect_agreement/agreement}")
    return texts_to_keep
           
                    
                
        

def get_tokens(word, model):
    tokenizer_with_prefix_space = AutoTokenizer.from_pretrained(model, add_prefix_space=True)
    return tokenizer_with_prefix_space([word], add_special_tokens=False).input_ids[0]
            
def evaluate_single_text(text, model='Llama-3.1-70B-Instruct', client = None, multi_label=False, use_together=False, logit_bias=False, neutral_option=True, scale_100 = False):

    ## DEFINE SYSTEM PROMPT
    if neutral_option:
        if scale_100:
        
            system_prompt = """
            Score the following post on a continual scale from 0 (left-wing) to 100 (right-wing) 
            where a score of 0 means "The post reflects radical left-wing ideology » 
            and a score of 100 means ' '"The post reflects radical right-wing ideology », 
            and a score of 50 means '"The post reflects a neutral ideology ».
            Reply with -1 if the post is not political. 
            Reply ONLY with the integer score (0-100 or -1). DO NOT reply with text.  
            Here is the text:"""
                
        else:
                
                system_prompt = """
                        You are a helpful assistant, expert in political lean classification. Your task is to classify the political lean of the text you receive. 
                        Output your answers as a label: 'L' for left-wing, 'R' for right-wing and 'N' for non-political. Do not output any other information than the label. Here is the text:
                        """
                lean_to_label = {"L": "Liberal", "R": "Conservative", "N": "Neutral"}

    else:
        if scale_100:
            system_prompt = """
            Score the following post on a continual scale from 0 (left-wing) to 100 (right-wing) 
            where a score of 0 means "The post reflects radical left-wing ideology » 
            and a score of 100 means ' '"The post reflects radical right-wing ideology », 
            and a score of 50 means '"The post reflects a neutral ideology ». 
            Reply ONLY with the integer score (0-100). DO NOT reply with text.  
            Here is the text:"""
        else:

            system_prompt = """
                    You are a helpful assistant, expert in political lean classification.
                    Your task is to classify the political lean of the text you receive.
                    Output your answers as a label: 'L' for left-wing and 'R' for right-wing. Do not output any other information than the label. Here is the text:
                    """
            lean_to_label = {"L": "Liberal", "R": "Conservative"}


    ## LOAD THE MODEL 

    if model == 'preloaded':
        model_id = "/lustre/fsn1/projects/rech/imi/utu57ed/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-70B-Instruct/snapshots/05917295788658563fd7ef778b6240ad9867d6d1/"
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        
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
            

            llm = AutoModelForCausalLM.from_pretrained(model_id, device_map = "auto", trust_remote_code=True, torch_dtype = torch.bfloat16).eval()
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            tokenizer.pad_token_id = tokenizer.eos_token_id  # Set a padding token

            
        
       


    elif model == 'gpt-4o':

        from openai import AzureOpenAI
        client = AzureOpenAI(
                            azure_endpoint=os.getenv(f"AZURE_OPENAI_ENDPOINT_gpt_4o_0513"),
                            api_key=os.getenv("AZURE_OPENAI_KEY_gpt_4o_0513"),
                            api_version=os.getenv("AZURE_OPENAI_API_VERSION_gpt_4o_0801"),
                        )
        

        


            

    else:
        print(f"Classifier {model} not recognized")
        return
    
    ## Add logi bias 
    if logit_bias:
        sequence_bias = [[get_tokens("L", model=model_id), 100.0], [get_tokens("R", model=model_id), 100.0]]
    else:
        sequence_bias = None
    
    ## CLASSIFY THE TEXT

    if model == 'preloaded':

        message_text = [{"role": "system", "content": system_prompt}, {"role": "user", "content": text}]
        output = client.chat.completions.create(
            model="llama",
            temperature=0.01,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            max_tokens=1,
            logprobs=True,
            top_logprobs=20,
        )

        #predicted_label = lean_to_label[output['labels'][0]]
        if scale_100:
            answers = ['L', 'R', 'N']
            _, generation, lprob, prob, leans = parse_hf_outputs(output=output, answers=answers, tokenizer=tokenizer, neutral_option=neutral_option)

        else:
                
            if neutral_option:
                        answers = ['L', 'R', 'N']
            else:
                answers = ['L', 'R']
            _, generation, lprob, prob, leans = parse_hf_outputs(output=output, answers=answers, tokenizer=tokenizer, neutral_option=neutral_option)

 
    elif model == 'gpt-4o':
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
        predicted_label = lean_to_label[res.choices[0].message.content]

    
    elif model == 'Llama-3.1-8B-Instruct' or model == 'Llama-3.1-70B-Instruct' or "Qwen2.5-72B-Instruct-Turbo":
        if use_together:
            res = client.chat.completions.create(
                        model=model_id,
                        temperature=0,
                        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": text}]
                        )
            predicted_label = lean_to_label[res.choices[0].message.content]
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            llm = AutoModelForCausalLM.from_pretrained(model_id, device_map = "auto", trust_remote_code=True, torch_dtype = torch.bfloat16).eval()
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            tokenizer.pad_token_id = tokenizer.eos_token_id  # Set a padding token
            tokens = tokenizer.apply_chat_template([system_prompt + text], add_generation_prompt=True, tokenize=False)
            inputs = tokenizer(tokens, padding=True, return_tensors="pt").to(llm.device)
            output = llm.generate(
                **inputs,
                max_new_tokens=1,
                return_dict_in_generate=True,
                output_scores=True,
                sequence_bias=sequence_bias,
                use_cache=True,
            )
            predicted_label = lean_to_label[output['labels'][0]]

            if neutral_option:
                        answers = ['L', 'R', 'N']
            else:
                answers = ['L', 'R']
            score, generation, lprob, prob, leans = parse_hf_outputs(output=output, answers=answers, tokenizer=tokenizer, neutral_option=neutral_option)

    return leans, generation, lprob, prob


        
        

                             
            

def evaluate_accuracy(dataset, n_samples=1000, batch_size=1, model='zero_shot', multi_label=False, seed = 42, use_together=False, logit_bias = False, neutral_option = True):


    if model == 'Llama-3.1-8B-Instruct' or model == 'Llama-3-70B-Instruct' or model == 'Llama-3.1-70B-Instruct' or  model == 'Llama-3.1-70B-Instruct-Turbo' or model == 'Mixtral-8x22B-Instruct-v0.1' or model == 'Mixtral-8x7B-Instruct-v0.1' or model == 'Mistral-7B-Instruct-v0.1' or model == 'Qwen2.5-72B-Instruct-Turbo':
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
            

            llm = AutoModelForCausalLM.from_pretrained(model_id, device_map = "auto", trust_remote_code=True, torch_dtype = torch.bfloat16).eval()
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
   
    elif model == 'zero_shot':
        # zero shot
        political_nlp = pipeline(model="facebook/bart-large-mnli", task="zero-shot-classification")
        hypothesis_template = "The auhtor of this text is a {}."
        if multi_label:
            candidate_labels = ['Left-wing', 'Right-wing', 'Democrat', 'Republican', 'Liberal', 'Conservative']
        else:
            candidate_labels = ["Republican", "Democrat"]
        party_to_label = {"Republican": "Conservative", "Democrat": "Liberal", "Left-wing": "Liberal", "Right-wing": "Conservative", "Conservative": "Conservative", "Liberal": "Liberal"}
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

        

    else:
        print(f"Classifier {model} not recognized")
        return
    
    ## Add logi bias 
    if logit_bias:
        sequence_bias = [[get_tokens("L", model=model_id), 100.0], [get_tokens("R", model=model_id), 100.0]]
    else:
        sequence_bias = None
    
    dataset_names = []
    if dataset == "all":
        dataset_names = ["twitter", "reddit", "reddit2", "news"]
    else:
        dataset_names = [dataset]
    for dataset_name in dataset_names:
        data = load_human_dataset(dataset_name)
        print(f"Loaded {len(data)} samples from {dataset_name}")
        
        # Shuffle the data to ensure random sampling
        # random.shuffle(data)  # You can also use sklearn's shuffle(data) if needed
        
        
        correct_left = 0
        correct_right = 0
        correct = 0
        left_samples = 0
        right_samples = 0
        
        # Add counters for precision/recall calculation
        true_positive_left = 0
        false_positive_left = 0
        false_negative_left = 0

        true_positive_right = 0
        false_positive_right = 0
        false_negative_right = 0

        texts = [row['text'] for row in data]
        labels = [row['Political Lean'] for row in data]

        # Shuffle the data to ensure random sampling, while keeping mapping between texts and labels
        texts, labels = shuffle(texts, labels, random_state=seed)  

        if model == 'gpt-4o':
            ## Provide cost estimation
            mean_length = sum([len(text) for text in texts])/len(texts)
            mean_input_char = len(system_prompt) + mean_length
            mean_input_tokens = mean_input_char/4
            input_price_per_token = 2.50 / 1000000

            mean_output_char = 1
            mean_output_tokens = mean_output_char
            output_price_per_token = 10.00 / 1000000

            mean_cost = (mean_input_tokens * input_price_per_token + mean_output_tokens * output_price_per_token) * 2 * n_samples

            print(f"{model}: Mean cost for {n_samples} samples of each lean: {mean_cost} $")
            input("Press enter to continue")
        
        elif model == 'gpt-4o-mini':
            ## Provide cost estimation
            mean_length = sum([len(text) for text in texts])/len(texts)
            mean_input_char = len(system_prompt) + mean_length
            mean_input_tokens = mean_input_char/4
            input_price_per_token = 0.150 / 1000000

            mean_output_char = 1
            mean_output_tokens = mean_output_char/4
            output_price_per_token = 0.600  / 1000000

            mean_cost = (mean_input_tokens * input_price_per_token + mean_output_tokens * output_price_per_token) * 2 * n_samples

            print(f"{model}: Mean cost for {n_samples} samples of each lean: {mean_cost} $")
            input("Press enter to continue")

       

        #
        
        # Process in batches
        data_to_store = { "texts": [], "predcited_labels": [], "actual_labels": [], "scores": [], "leans": [], "lprobs": [], "probs": []}
        for i in trange(0, len(data), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]

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

                    for text in batch_texts:
                    
                        res = client.chat.completions.create(
                        model=model_id,
                        temperature=0,
                        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": text}]
                        )
                        results.append(res.choices[0].message.content)

                else:
                    conversations = [[{"role": "system", "content": system_prompt}, {"role": "user", "content": text}] for text in batch_texts]

                    # texts = tokenizer.apply_chat_template([[system_prompt + text for text in batch_texts]])
                    # output = llm.generate(inputs["input_ids"],return_dict_in_generate=True, max_new_tokens=1, sequence_bias=sequence_bias, output_scores=True)
                    print("applying chat template")
                    tokens = tokenizer.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)
                    print('tokenizing')

                    # inputs = tokenizer(tokens, return_tensors="pt")

                    inputs = tokenizer(tokens, padding=True, return_tensors="pt").to(llm.device)

                    print(f"generating batch {i} of {len(data)}")

                    print(inputs)

                    # from IPython import embed
                    # embed()
                    output = llm.generate(
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
                #batch_results = [{"labels": [res]} for res in results]
            
            elif model == 'PoliBERT':
                results = []
                for text in batch_texts:
                    example = "A poltician from the <mask> party said: " + text
                    fill_mask = pipeline('fill-mask', model=pretrained_LM_path, tokenizer=political_tokenizer)
                    outputs = fill_mask(example)
                    #find most probable word between democratic and Republican
                    for output in outputs:
                        #remove capitalization
                        output['token_str'] = output['token_str'].lower()
                        if output['token_str'] == 'democratic' or output['token_str'] == 'republican':
                            break

                    results.append(output['token_str'])
                batch_results = [{"labels": [res]} for res in results]

            # Iterate through batch results
            for j in range(len(batch_results)):
                actual_label = batch_labels[j]
                #check that samples are still needed
                if left_samples >= n_samples and actual_label == 'Liberal' or right_samples >= n_samples and actual_label == 'Conservative':
                    continue

                # print(f"Left samples: {left_samples}, Right samples: {right_samples}")
                # print(f"actual label: {batch_labels[j]}")
                if model == 'zero_shot':
                    # predicted_label = batch_results[j]['labels'][0]
                    predicted_label = party_to_label[batch_results[j]['labels'][0]]
                elif model == 'DeBertA':
                    predicted_label = lean_to_label[batch_results[j][0]['label']]
                    #if predictedf label is center, take the second highest score
                    if predicted_label == 'Neutral':
                        predicted_label = lean_to_label[batch_results[j][1]['label']]

                

                elif model == 'gpt-4o' or model == 'gpt-4o-mini':
                    try:
                        predicted_label = lean_to_label[batch_results[j]['labels'][0]]
                    except:
                        print(f"Error in LLM output: {batch_results[j]}")
                        predicted_label = "Neutral"
                
                elif model == 'Llama-3.1-8B-Instruct' or model == 'Llama-3.1-70B-Instruct' or "Qwen2.5-72B-Instruct-Turbo":

                    try:
                        predicted_label = lean_to_label[batch_results[0]['labels'][0][j]]
                    except:
                        #embded:
                        # from IPython import embed
                        # embed()
                        print(f"Error in LLM output: {batch_results[j]}")
                        predicted_label = "Neutral"

                elif model == 'PoliBERT':
                    try:
                        predicted_label = lean_to_label[batch_results[j]['labels'][0]]
                    except:
                        predicted_label = "Neutral"



                

                # print(f"predicted label: {predicted_label}")

                
                if predicted_label == actual_label:
                    correct += 1
                
                if actual_label == "Conservative":
                    right_samples += 1
                    if predicted_label == actual_label:
                        correct_right += 1
                        true_positive_right += 1
                    else:
                        false_negative_right += 1
                        if predicted_label == "Liberal":
                            false_positive_left += 1

                elif actual_label == "Liberal":
                    left_samples += 1
                    if predicted_label == actual_label:
                        correct_left += 1
                        true_positive_left += 1
                    else:
                        false_negative_left += 1
                        if predicted_label == "Conservative":
                            false_positive_right += 1
                
                # Store data for later analysis
                data_to_store["texts"].append(batch_texts[j])
                data_to_store["predcited_labels"].append(predicted_label)
                data_to_store["actual_labels"].append(actual_label)

                if model == 'zero_shot':
                    data_to_store["scores"].append(batch_results[j]['scores'][0])
                elif model == 'DeBertA':
                    data_to_store["scores"].append(batch_results[j][0]['score'])
                elif model == 'gpt-4o' or model == 'gpt-4o-mini':
                    data_to_store["scores"].append(None)
                elif model == 'Llama-3.1-8B-Instruct-Turbo' or model == 'Llama-3.1-70B-Instruct-Turbo' or "Qwen2.5-72B-Instruct-Turbo":
                    data_to_store["leans"].append(batch_results[0]['leans'][j])
                    # data_to_store["scores"].append(batch_results[0]['scores'][0][0])
                    data_to_store["lprobs"].append(batch_results[0]['lprobs'][0][j])
                    # data_to_store["probs"].append(batch_results[0]['probs'][0][0])

                    # print("Label: ",predicted_label)
                    # print("Score: ", batch_results[j]['scores'][0])

                # Stop when we reach the required number of left and right samples
                if left_samples >= n_samples and right_samples >= n_samples:
                    break

            # Stop if enough samples have been processed
            if left_samples >= n_samples and right_samples >= n_samples:
                break
            else:
                pass
                # print(f"Left samples: {left_samples}, Right samples: {right_samples}, n_samples: {n_samples}")

        # Calculate precision and recall for both classes
        precision_left = true_positive_left / (true_positive_left + false_positive_left) if (true_positive_left + false_positive_left) > 0 else 0
        recall_left = true_positive_left / (true_positive_left + false_negative_left) if (true_positive_left + false_negative_left) > 0 else 0

        precision_right = true_positive_right / (true_positive_right + false_positive_right) if (true_positive_right + false_positive_right) > 0 else 0
        recall_right = true_positive_right / (true_positive_right + false_negative_right) if (true_positive_right + false_negative_right) > 0 else 0

       
        print(f"Precision for 'Liberal': {precision_left}")
        print(f"Recall for 'Liberal': {recall_left}")
        print(f"Precision for 'Conservative': {precision_right}")
        print(f"Recall for 'Conservative': {recall_right}")

        #create a text file with the results
        classifier = model
        classifier = f"{classifier}_multi_label" if multi_label else classifier
        with open(f"results_{dataset_name}_{classifier}_{n_samples}samples_seed{seed}.txt", "w") as f:
            f.write(f'Classifier: {classifier}\n') 
            f.write(f"Precision for 'Liberal': {precision_left}\n")
            f.write(f"Recall for 'Liberal': {recall_left}\n")
            f.write(f"Precision for 'Conservative': {precision_right}\n")
            f.write(f"Recall for 'Conservative': {recall_right}\n")
        
        #save results to pickle file
        with open(f"results_{dataset_name}_{classifier}_{n_samples}samples_seed{seed}.pkl", "wb") as f:
            pickle.dump({"precision_left": precision_left, "recall_left": recall_left, "precision_right": precision_right, "recall_right": recall_right}, f)
        
        #save data to pickle file
        with open(f"data_{dataset_name}_{classifier}_{n_samples}samples_seed{seed}.pkl", "wb") as f:
            pickle.dump(data_to_store, f)

            


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

def preditct_political_lean(batch_texts, model, political_nlp, system_prompt, candidate_labels, tokenizer , client = None, neutral_option = True, use_together = False, logit_bias = False):
    
    
    
    
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

            # from IPython import embed
            # embed()
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

           
        

def plot_from_pickle(n_samples, datasets = ['twitter'], models = ['zero_shot', 'DeBertA', 'gpt-4o', 'gpt-4o-mini'], seeds = [42]):
    import matplotlib.pyplot as plt
    import pickle
    import os
    import numpy as np
    import seaborn as sns
    import pandas as pd


    for dataset in datasets:
        data = []
        for model in models:
            #iterate over all seeds
            for seed in seeds:
                with open(f"results_{dataset}_{model}_{n_samples}samples_seed{seed}.pkl", "rb") as f:
                    results = pickle.load(f)
                    precision_left = results['precision_left']
                    recall_left = results['recall_left']
                    precision_right = results['precision_right']
                    recall_right = results['recall_right']

                    data.append({"model": model, "value": precision_left, "type": "precision", "lean": 'left', "seed": seed})
                    data.append({"model": model, "value": recall_left, "type": "recall", "lean": 'left', "seed": seed})
                    data.append({"model": model, "value": precision_right, "type": "precision", "lean": 'right', "seed": seed})
                    data.append({"model": model, "value": recall_right, "type": "recall", "lean": 'right', "seed": seed})

        
        fig = plt.figure()
        #Increase figure size
        fig.set_size_inches(10, 6)
        df = pd.DataFrame(data)
        sns.barplot(x='model', y='value', hue='type', data=df[df['lean'] == 'left'])
        

        plt.legend()
        plt.title(f"Results for {dataset} dataset - Left lean")
        plt.show()
        fig.savefig(f"results_{dataset}_left_{n_samples}.png")

        fig = plt.figure()
        # Increase figure size
        fig.set_size_inches(10, 6)
        df = pd.DataFrame(data)
        sns.barplot(x='model', y='value', hue='type', data=df[df['lean'] == 'right'])

        plt.legend()
        plt.title(f"Results for {dataset} dataset - Right lean")
        plt.show()
        fig.savefig(f"results_{dataset}_right_{n_samples}.png")





    


#main 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agreement", action="store_true")
    parser.add_argument("--dataset_name", type=str, default="all")
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=1)
    # parser.add_argument("--zero_shot", action="store_true")
    parser.add_argument("--model", type=str, default="all")
    parser.add_argument("--seed", type=int, default=42)

    

    args = parser.parse_args()

    if args.agreement:
        classifier_agreement(args.dataset_name, args.n_samples, args.batch_size)
    else:
        if args.model == 'all':
            # evaluate_accuracy(args.dataset_name, args.n_samples, args.batch_size, model = 'zero_shot')
            # evaluate_accuracy(args.dataset_name, args.n_samples, args.batch_size, model='DeBertA', seed=args.seed)
            # evaluate_accuracy(args.dataset_name, args.n_samples, args.batch_size, model = 'zero_shot',  multi_label=True, seed=args.seed)
            # evaluate_accuracy(args.dataset_name, args.n_samples, args.batch_size, model='gpt-4o', seed=args.seed)
            # evaluate_accuracy(args.dataset_name, args.n_samples, args.batch_size, model='gpt-4o-mini', seed=args.seed)
            # evaluate_accuracy(args.dataset_name, args.n_samples, args.batch_size, model='PoliBERT', seed=args.seed)
            # evaluate_accuracy(args.dataset_name, args.n_samples, args.batch_size, model='Llama-3.1-8B-Instruct-Turbo', seed=args.seed)
            # evaluate_accuracy(args.dataset_name, args.n_samples, args.batch_size, model='Llama-3.1-70B-Instruct-Turbo', seed=args.seed)
            # evaluate_accuracy(args.dataset_name, args.n_samples, args.batch_size, model='Qwen2.5-72B-Instruct-Turbo', seed=args.seed)
            # plot_from_pickle(args.n_samples, models=['zero_shot', 'DeBertA', 'zero_shot_multi_label', 'gpt-4o', 'gpt-4o-mini', 'Llama-3.1-8B-Instruct-Turbo', 'Llama-3.1-70B-Instruct-Turbo', 'Qwen2.5-72B-Instruct-Turbo'], seeds=[args.seed])
            plot_from_pickle(args.n_samples, models=['zero_shot', 'DeBertA', 'zero_shot_multi_label', 'gpt-4o', 'gpt-4o-mini', 'Llama-3.1-8B-Instruct-Turbo', 'Llama-3.1-70B-Instruct-Turbo'])
        else:
            evaluate_accuracy(args.dataset_name, args.n_samples, args.batch_size, model=args.model)
