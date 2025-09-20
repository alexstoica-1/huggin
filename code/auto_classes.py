from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline
)

my_model = AutoModelForSequenceClassification.from_pretrained(
    'distilbert-base-uncased-finetuned-sst-2-english'
)

# use tokenizers paired with the models, you have to handle manually

my_tokenizer = AutoTokenizer.from_pretrained(
    'distilbert-base-uncased-finetuned-sst-2-english'
)

#tokens = tokenizer.tokenize("Alex is going to be the greatest AI scientist ever.")
#print(tokens)

pipeline = pipeline(
    task = 'sentiment-analysis',
    model = my_model,
    tokenizer = my_tokenizer
)

text = "Alex is going to be the greatest AI scientist ever."
result = pipeline(text)
print(f"The label is: {result[0]['label']} with score: {result[0]['score']}.")