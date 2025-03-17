from eval_utils import llama_pol_lean, llama_pol_lean_scale
from dataset_utils import *
from datasets import load_dataset, load_from_disk, concatenate_datasets
from datasets import Features, Value
from datetime import datetime

def remove_links(batch):
    return {"text": [re.sub(r'http\S+', '', t).rstrip() for t in batch['text']]}

def remove_trailling_hashtags(batch):
    return {"text": [re.sub(r"(?:\s*#\w+)+$", "", t) for t in batch['text']]}


def filter_posts_by_size(dataset, n_min = 20, n_max = 30):

    import tiktoken
    tokenizer = tiktoken.get_encoding("cl100k_base")
    dataset = dataset.map(
        lambda batch: {"n_tokens": [len(tokenizer.encode(t)) for t in batch['text']]},
        batched=True, desc="Adding number of tokens"
    )

    dataset = dataset.filter(
        lambda batch: [n_min < n_t < n_max for n_t in batch['n_tokens']],
        batched=True, desc=f"Filtering too long (>{n_max}) and too shorts (<{n_min}) posts."
    )

    #get only text where column 'speaker' is not empty
    dataset = dataset.filter(
        lambda batch: [len(t) > 0 for t in batch['speaker']],
        batched=True, desc="Filtering empty speakers"
    )
    

    return dataset

def is_before_gpt3_release(date_str):
    gpt3_release_date = datetime(2020, 6, 1)
    input_date = datetime.fromisoformat(date_str).replace(tzinfo=None)
    return input_date < gpt3_release_date


def clean_dataset(dataset, n_min = 20, n_max = 200):
    dataset = dataset.map(remove_links, batched=True, desc="Removing links", load_from_cache_file=False)
    dataset = dataset.filter(lambda examples: [len(word_tokenize(t)) > 10 for t in examples['text']], batched=True,
                            load_from_cache_file=False)
    # dataset = dataset.remove_columns(["embeddings"])
    # dataset = dataset.map(remove_links, batched=True, desc="Removing links", load_from_cache_file=True, num_proc=10)
    # dataset = dataset.map(remove_trailling_hashtags, batched=True, desc="Removing links", load_from_cache_file=True, num_proc=10)
    # dataset = filter_posts_by_size(dataset, n_min=n_min, n_max=n_max)
    
    return dataset





try:
    senator_dataset = load_from_disk(f"./data/senator_tweets/prepared-political-dataset-scale") 
    print('loaded')
except:


    try:
        senator_dataset = load_from_disk(f"./data/senator_tweets/prepared-political-dataset-cleaned")
        
        #add placeholder column
        senator_dataset['train'] = senator_dataset['train'].map(
            lambda examples: {"llama_pol_lean_scale": None},
            desc="Adding placeholder column"
        )

        senator_dataset['test'] = senator_dataset['test'].map(
            lambda examples: {"llama_pol_lean_scale": None},
            desc="Adding placeholder column"
        )

        # Define the updated schema
        new_features = senator_dataset['train'].features.copy()
        new_features["llama_pol_lean_scale"] = Value("float64")

        # Cast the dataset to the new schema
        senator_dataset['train'] = senator_dataset['train'].cast(new_features)
        senator_dataset['test'] = senator_dataset['test'].cast(new_features)


        senator_dataset['train'] = senator_dataset['train'].map(
            lambda examples: {"llama_pol_lean_scale": llama_pol_lean_scale(examples["text"])},
            batched=True, desc="Computing Political lean", batch_size=10, num_proc=30
        )

        senator_dataset['test'] = senator_dataset['test'].map(
            lambda examples: {"llama_pol_lean_scale": llama_pol_lean_scale(examples["text"])},
            batched=True, desc="Computing Political lean", batch_size=10, num_proc=30
        ) 

        #save cleaned dataset
        senator_dataset.save_to_disk(f"./data/senator_tweets/prepared-political-dataset-scale")
    
    except:
    

            

        senator_dataset = load_dataset("m-newhauser/senator-tweets")

        #clean dataset
        senator_dataset = clean_dataset(senator_dataset)
            
        #add placeholder column
        senator_dataset['train'] = senator_dataset['train'].map(
            lambda examples: {"llama_pol_lean": None},
            desc="Adding placeholder column"
        )

        senator_dataset['test'] = senator_dataset['test'].map(
            lambda examples: {"llama_pol_lean": None},
            desc="Adding placeholder column"
        )

        # Define the updated schema
        new_features = senator_dataset['train'].features.copy()
        new_features["llama_pol_lean"] = Value("float64")

        # Cast the dataset to the new schema
        senator_dataset['train'] = senator_dataset['train'].cast(new_features)
        senator_dataset['test'] = senator_dataset['test'].cast(new_features)


        senator_dataset['train'] = senator_dataset['train'].map(
            lambda examples: {"llama_pol_lean": llama_pol_lean(examples["text"])},
            batched=True, desc="Computing Political lean", batch_size=10, num_proc=30
        )

        senator_dataset['test'] = senator_dataset['test'].map(
            lambda examples: {"llama_pol_lean": llama_pol_lean(examples["text"])},
            batched=True, desc="Computing Political lean", batch_size=10, num_proc=30
        )    

         #add placeholder column
        senator_dataset['train'] = senator_dataset['train'].map(
            lambda examples: {"llama_pol_lean": None},
            desc="Adding placeholder column"
        )

        senator_dataset['test'] = senator_dataset['test'].map(
            lambda examples: {"llama_pol_lean_scale": None},
            desc="Adding placeholder column"
        )

        # Define the updated schema
        new_features = senator_dataset['train'].features.copy()
        new_features["llama_pol_lean_scale"] = Value("float64")

        # Cast the dataset to the new schema
        senator_dataset['train'] = senator_dataset['train'].cast(new_features)
        senator_dataset['test'] = senator_dataset['test'].cast(new_features)


        senator_dataset['train'] = senator_dataset['train'].map(
            lambda examples: {"llama_pol_lean_scale": llama_pol_lean_scale(examples["text"])},
            batched=True, desc="Computing Political lean", batch_size=10, num_proc=30
        )

        senator_dataset['test'] = senator_dataset['test'].map(
            lambda examples: {"llama_pol_lean_scale": llama_pol_lean_scale(examples["text"])},
            batched=True, desc="Computing Political lean", batch_size=10, num_proc=30
        ) 

        senator_dataset.save_to_disk(f"./data/senator_tweets/prepared-political-dataset-scale") 


# try: 
#     reddit_dataset = load_from_disk(f"./data/redditLibCon/prepared-political-dataset-cleaned")
#     print('loaded')
# except:

   




#     try:
#         reddit_dataset = load_from_disk(f"./data/redditLibCon/merged-dataset-200-minus-20-plus-cleaned")
#         print('loaded')
#     except:
        

#         reddit_dataset_conservative = load_from_disk("./data/redditLibCon/Conservative")
#         reddit_dataset_liberal = load_from_disk("./data/redditLibCon/Liberal")

#         reddit_dataset = concatenate_datasets([reddit_dataset_conservative, reddit_dataset_liberal])

        
#         reddit_dataset = reddit_dataset.rename_column("articles", "text")




#         n_max = 300
#         n_min = 20
        
#         reddit_dataset = clean_dataset(reddit_dataset, n_min=n_min, n_max=n_max)
        
#         #save to disk
#         reddit_dataset.save_to_disk(f"./data/redditLibCon/merged-dataset-{n_max}-minus-{n_min}-plus-cleaned")
#         print("Cleaned reddit dataset size:", len(reddit_dataset))
#     #add placeholder column 
#     reddit_dataset = reddit_dataset.map(
#         lambda examples: {"llama_pol_lean": None},
#         desc="Adding placeholder column"
#     )

#     # Define the updated schema
#     new_features = reddit_dataset.features.copy()
#     new_features["llama_pol_lean"] = Value("float64")

#     # Cast the dataset to the new schema
#     reddit_dataset = reddit_dataset.cast(new_features)


    
#     reddit_dataset = reddit_dataset.map(
#         lambda examples: {"llama_pol_lean": llama_pol_lean(examples["text"])},
#         batched=True, desc="Computing Political lean", batch_size=10, num_proc=30
#     )

#     reddit_dataset.save_to_disk(f"./data/redditLibCon/prepared-political-dataset-cleaned")

# try:
#     reddit_dataset_lp = load_from_disk(f"./data/redditLibCon/prepared-left-polarization-political-dataset")
# except:
#     #save split datasets
#     reddit_dataset_lp = reddit_dataset.filter(lambda ex: ex['llama_pol_lean'] < 0)
#     reddit_dataset_lp.save_to_disk(f"./data/redditLibCon/prepared-left-polarization-political-dataset")
#     print('reddit_dataset_lp size:', len(reddit_dataset_lp))

# try:
#     reddit_dataset_rp = load_from_disk(f"./data/redditLibCon/prepared-right-polarization-political-dataset")
# except:
#     reddit_dataset_rp = reddit_dataset.filter(lambda ex: ex['llama_pol_lean'] > 0)
#     reddit_dataset_rp.save_to_disk(f"./data/redditLibCon/prepared-right-polarization-political-dataset")
#     print('reddit_dataset_rp size:', len(reddit_dataset_rp))

# try:
#     reddit_dataset_np = load_from_disk(f"./data/redditLibCon/prepared-no-polarization-political-dataset")
# except:
#     reddit_dataset_np = reddit_dataset.filter(lambda ex: ex['llama_pol_lean'] == 0)
#     reddit_dataset_np.save_to_disk(f"./data/redditLibCon/prepared-no-polarization-political-dataset")
#     print('reddit_dataset_np size:', len(reddit_dataset_np))





# n_min = 20
# n_max = 200


# dataset_100m = load_from_disk(f"./data/twitter_100m/prepared-100m-tweets-english-qualities-before-gpt3-{n_max}-minus-{n_min}-plus")

# try:
#     dataset_100m_poltical = load_from_disk(f"./data/twitter_100m/prepared-100m-tweets-english-qualities-before-gpt3-{n_max}-minus-{n_min}-political")
#     print('loaded')

    


    
# except:
#     #Add llama_pol_lean to dataset2

   

#     dataset_100m = dataset_100m.map(
#     lambda examples: {"llama_pol_lean": None},
#     desc="Adding placeholder column"
#     )   

#     # Define the updated schema
#     new_features = dataset_100m.features.copy()
#     new_features["llama_pol_lean"] = Value("float64")

#     # Cast the dataset to the new schema
#     dataset_100m_small = dataset_100m.cast(new_features)


#     dataset_100m_poltical = dataset_100m.map( 
#         lambda examples: {"llama_pol_lean": llama_pol_lean(examples["text"])},
#         batched=True, desc="Computing Political lean", batch_size=10, num_proc=30
#     )
#     dataset_100m_poltical.save_to_disk(f"./data/twitter_100m/prepared-100m-tweets-english-qualities-before-gpt3-{n_max}-minus-{n_min}-political-full")





# try:
#     dataset_webis_poltical = load_from_disk(f"./data/webis/prepared-political-quality-diversity-no-tldr-200-minus-20-plus-clear-corpus-webis-tldr-17")
#     print('loaded')

    


    
# except:
#     #Add llama_pol_lean to dataset2

#     dataset_webis = load_from_disk("/lustre/fsn1/projects/rech/imi/utu57ed/Projects/CE_LLMS_project/CE_llms/data/webis/prepared-quality-no-tldr-200-minus-20-plus-clear-corpus-webis-tldr-17")

#     #only keep politics subreddit
#     dataset_webi_train_politics = dataset_webis['train'].filter(lambda ex: ex['subreddit'] == 'politics')
#     dataset_webi_test_politics = dataset_webis['test'].filter(lambda ex: ex['subreddit'] == 'politics')

#     #merge 
#     dataset_webis = concatenate_datasets([dataset_webi_train_politics, dataset_webi_test_politics])

#     #rstrip
#     dataset_webis = dataset_webis.map(lambda examples: {"text": [t.rstrip() for t in examples["text"]]}, batched=True, desc="rstrip", num_proc=64)

   

#     dataset_webis = dataset_webis.map(
#     lambda examples: {"llama_pol_lean": None},
#     desc="Adding placeholder column"
#     )   

#     # Define the updated schema
#     new_features = dataset_webis.features.copy()
#     new_features["llama_pol_lean"] = Value("float64")

#     # Cast the dataset to the new schema
#     dataset_100m_small = dataset_webis.cast(new_features)


#     dataset_webis_poltical = dataset_webis.map( 
#         lambda examples: {"llama_pol_lean": llama_pol_lean(examples["text"])},
#         batched=True, desc="Computing Political lean", batch_size=10, num_proc=30
#     )
#     dataset_webis_poltical.save_to_disk(f"./data/webis/prepared-political-quality-diversity-no-tldr-200-minus-20-plus-clear-corpus-webis-tldr-17")
    


# dataset = dataset_webis_poltical

# try:
#     split_dataset_lp = load_from_disk(f"./data/webis/prepared-left-polarization-political-dataset-cleaned")
#     print('loaded split_dataset_lp')
# except:

#     #Save left-polarization (lp)
#     split_dataset_lp = dataset.filter(lambda ex: ex['llama_pol_lean'] < 0)
#     split_dataset_lp.save_to_disk(f"./data/webis/prepared-left-polarization-political-dataset-cleaned")
#     #print size
#     print('split_dataset_lp size:', len(split_dataset_lp))

# try: 
#     split_dataset_rp = load_from_disk(f"./data/webis/prepared-right-polarization-political-dataset-cleaned")
#     print('loaded split_dataset_rp')
# except:
#     #Save right-polarization (rp)
#     split_dataset_rp = dataset.filter(lambda ex: ex['llama_pol_lean'] > 0)
#     split_dataset_rp.save_to_disk(f"./data/webis/prepared-right-polarization-political-dataset-cleaned")
#     #print size
#     print('split_dataset_rp size:', len(split_dataset_rp)) 

# try:
#     split_dataset_np = load_from_disk(f"./data/webis/prepared-no-polarization-political-dataset-cleaned")
#     print('loaded split_dataset_np')
# except:
#     #Save no-polarization (np)
#     split_dataset_np = dataset.filter(lambda ex: ex['llama_pol_lean'] == 0)
#     split_dataset_np.save_to_disk(f"./data/webis/prepared-no-polarization-political-dataset-cleaned")
#     #print size
#     print('split_dataset_np size:', len(split_dataset_np))








#merge datasets
senator_dataset = concatenate_datasets([senator_dataset['train'], senator_dataset['test']])



dataset = senator_dataset

try:
    split_dataset_lp = load_from_disk(f"./data/senator_tweets_and_100m/prepared-left-polarization-political-dataset-cleaned")
    print('loaded split_dataset_lp')
except:

    #Save left-polarization (lp)
    split_dataset_lp = dataset.filter(lambda ex: ex['llama_pol_lean'] < 0)
    split_dataset_lp.save_to_disk(f"./data/senator_tweets_and_100m/prepared-left-polarization-political-dataset-cleaned")
    #print size
    print('split_dataset_lp size:', len(split_dataset_lp))

try: 
    split_dataset_rp = load_from_disk(f"./data/senator_tweets_and_100m/prepared-right-polarization-political-dataset-cleaned")
    print('loaded split_dataset_rp')
except:
    #Save right-polarization (rp)
    split_dataset_rp = dataset.filter(lambda ex: ex['llama_pol_lean'] > 0)
    split_dataset_rp.save_to_disk(f"./data/senator_tweets_and_100m/prepared-right-polarization-political-dataset-cleaned")
    #print size
    print('split_dataset_rp size:', len(split_dataset_rp)) 

try:
    split_dataset_np = load_from_disk(f"./data/senator_tweets_and_100m/prepared-no-polarization-political-dataset-cleaned")
    print('loaded split_dataset_np')
except:
    #Save no-polarization (np)
    split_dataset_np = dataset.filter(lambda ex: ex['llama_pol_lean'] == 0)
    split_dataset_np.save_to_disk(f"./data/senator_tweets_and_100m/prepared-no-polarization-political-dataset-cleaned")
    #print size
    print('split_dataset_np size:', len(split_dataset_np))



##### Save mixed datasets; the size of resulting datasets should be the same as the size of the smallest split dataset

max_size = min(len(split_dataset_lp), len(split_dataset_rp))

#Save 50-50-polarization (50l50r)
split_dataset_50l50r = concatenate_datasets([split_dataset_lp.select(range(max_size//2)), split_dataset_rp.select(range(max_size//2))])
split_dataset_50l50r.save_to_disk(f"./data/senator_tweet/prepared-50-50-polarization-political-dataset")
print('split_dataset_50l50r size:', len(split_dataset_50l50r))

#Save 25-75-polarization (25l75r)
split_dataset_25l75r = concatenate_datasets([split_dataset_lp.select(range(max_size//4)), split_dataset_rp.select(range(3*max_size//4))])
split_dataset_25l75r.save_to_disk(f"./data/senator_tweet/prepared-25-75-polarization-political-dataset")
print('split_dataset_25l75r size:', len(split_dataset_25l75r))

#Save 75-25-polarization (75l25r)
split_dataset_75l25r = concatenate_datasets([split_dataset_lp.select(range(3*max_size//4)), split_dataset_rp.select(range(max_size//4))])
split_dataset_75l25r.save_to_disk(f"./data/senator_tweet/prepared-75-25-polarization-political-dataset")
print('split_dataset_75l25r size:', len(split_dataset_75l25r))

#Save 0-100-polarization (0l100r)
split_dataset_0l100r = split_dataset_rp.select(range(max_size))
split_dataset_0l100r.save_to_disk(f"./data/senator_tweet/prepared-0-100-polarization-political-dataset")
print('split_dataset_0l100r size:', len(split_dataset_0l100r))

#Save 100-0-polarization (100l0r)
split_dataset_100l0r = split_dataset_lp.select(range(max_size))
split_dataset_100l0r.save_to_disk(f"./data/senator_tweet/prepared-100-0-polarization-political-dataset")
print('split_dataset_100l0r size:', len(split_dataset_100l0r))

