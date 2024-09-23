from transformers import pipeline
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
#     print(f"text: {text} -> {political_nlp(text)}")

# zero shot
political_nlp = pipeline(model="facebook/bart-large-mnli", task="zero-shot-classification")
hypothesis_template = "This is a post from a {}."
candidate_labels = ["democrat", "republican"]

for text in texts:
    res = political_nlp(text, candidate_labels=candidate_labels, hypothesis_template=hypothesis_template)
    print(f"text: {text} -> {res['labels'][0]} {res['scores'][0]}")

