import argparse
from tqdm import trange
from transformers import pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig

from dataset_utils import load_human_dataset
texts = [
    "Trans rights are the most important issue today!",
    "I am a left winger!",
    "I am a communist!",
    "I love Biden!",
    "I love the democrats!",
    "I love the republicans!",
    "I love the bible!",
    "I love Trump!",
    "Enough with the immigrants!",
    "If you're gay just stay away from me!",
    "I don't care if you're gay but stay away from me!",
    "I like apples.",
    "Happy New Year!"
]

# DeBertA
# label2id = {"Left": 0, "Center": 1, "Right": 2}
# id2label = {0: "Left", 1: "Center", 2: "Right"}
#
# config = AutoConfig.from_pretrained("premsa/political-bias-prediction-allsides-DeBERTa",label2id=label2id, id2label=id2label)
# political_model = AutoModelForSequenceClassification.from_pretrained("premsa/political-bias-prediction-allsides-DeBERTa", config=config)
# political_tokenizer = AutoTokenizer.from_pretrained("premsa/political-bias-prediction-allsides-DeBERTa")
#
# political_nlp = pipeline("text-classification", model=political_model, tokenizer=political_tokenizer)
#
# for text in texts:
# #     print(f"text: {text} -> {political_nlp(text)}")

# # zero shot
# political_nlp = pipeline(model="facebook/bart-large-mnli", task="zero-shot-classification")
# hypothesis_template = "The political lean of this text is {}."
# candidate_labels = ["Conservative", "Liberal"]

# for text in texts:
#     res = political_nlp(text, candidate_labels=candidate_labels, hypothesis_template=hypothesis_template)
#     print(f"text: {text} -> {res['labels'][0]} {res['scores'][0]}")





## Function to evaluate accuracy

import random
from sklearn.utils import shuffle  # You can use this if you prefer sklearn's shuffle

def evaluate_accuracy(dataset_name, n_samples=1000, batch_size=1, zero_shot=True):
    if zero_shot:
        # zero shot
        political_nlp = pipeline(model="facebook/bart-large-mnli", task="zero-shot-classification")
        hypothesis_template = "The auhtor of this text is a {}."
        candidate_labels = ["Republican", "Democrat"]
        party_to_label = {"Republican": "Conservative", "Democrat": "Liberal"}
    else:
        # DeBertA
        label2id = {"Left": 0, "Center": 1, "Right": 2}
        id2label = {0: "Left", 1: "Center", 2: "Right"}

        config = AutoConfig.from_pretrained("premsa/political-bias-prediction-allsides-DeBERTa", label2id=label2id, id2label=id2label)
        political_model = AutoModelForSequenceClassification.from_pretrained("premsa/political-bias-prediction-allsides-DeBERTa", config=config)
        political_tokenizer = AutoTokenizer.from_pretrained("premsa/political-bias-prediction-allsides-DeBERTa")

        political_nlp = pipeline("text-classification", model=political_model, tokenizer=political_tokenizer)
    
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

    # Shuffle the data to ensure random sampling
    # texts, labels = shuffle(texts, labels) 

    #
    
    # Process in batches
    for i in trange(0, len(data), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]

        if zero_shot:
            batch_results = political_nlp(batch_texts, candidate_labels=candidate_labels, hypothesis_template=hypothesis_template)
        else:
            batch_results = political_nlp(batch_texts)

        # Iterate through batch results
        for j in range(len(batch_results)):
            # print(f"Left samples: {left_samples}, Right samples: {right_samples}")
            # print(f"actual label: {batch_labels[j]}")
            if zero_shot:
                # predicted_label = batch_results[j]['labels'][0]
                predicted_label = party_to_label[batch_results[j]['labels'][0]]
            else:
                predicted_label = batch_results[j]['label']
            actual_label = batch_labels[j]
            
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

            # Stop when we reach the required number of left and right samples
            if left_samples >= n_samples and right_samples >= n_samples:
                break

        # Stop if enough samples have been processed
        if left_samples >= n_samples and right_samples >= n_samples:
            break

    # Calculate precision and recall for both classes
    precision_left = true_positive_left / (true_positive_left + false_positive_left) if (true_positive_left + false_positive_left) > 0 else 0
    recall_left = true_positive_left / (true_positive_left + false_negative_left) if (true_positive_left + false_negative_left) > 0 else 0

    precision_right = true_positive_right / (true_positive_right + false_positive_right) if (true_positive_right + false_positive_right) > 0 else 0
    recall_right = true_positive_right / (true_positive_right + false_negative_right) if (true_positive_right + false_negative_right) > 0 else 0

    print(f"Accuracy on {dataset_name} for {left_samples} left samples:  {correct_left/left_samples}")
    print(f"Accuracy on {dataset_name} for {right_samples} right samples:  {correct_right/right_samples}")
    print(f"Accuracy on {dataset_name} for all {left_samples+right_samples} samples: {correct/(left_samples+right_samples)}")
    
    print(f"Precision for 'Liberal': {precision_left}")
    print(f"Recall for 'Liberal': {recall_left}")
    print(f"Precision for 'Conservative': {precision_right}")
    print(f"Recall for 'Conservative': {recall_right}")
    
    return correct/(left_samples+right_samples), precision_left, recall_left, precision_right, recall_right


#main 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="all")
    parser.add_argument("--n_samples", type=int, default=1000)

    args = parser.parse_args()

    evaluate_accuracy(args.dataset_name, args.n_samples)


    
    




