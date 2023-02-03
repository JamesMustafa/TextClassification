import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

# Encode the input text
input_text = "Hello, my dog is cute"
inputs = tokenizer(input_text, return_tensors="pt")

# Generate the sentiment predictions
with torch.no_grad():
    outputs = model(**inputs)
    prediction = outputs[0].argmax().item()

# Map the prediction to a sentiment label
if prediction == 0:
    sentiment = "negative"
else:
    sentiment = "positive"

# Print the sentiment
print(f"The sentiment for '{input_text}' is: {sentiment}")
