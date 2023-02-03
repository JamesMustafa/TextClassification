import torch
import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# Get the comments from the response
def fetch_all_comments(video_id, api_key):
    url = "https://www.googleapis.com/youtube/v3/commentThreads"
    comments = []
    next_page_token = None
    while True:
        params = {
            "part": "snippet",
            "maxResults": 100,
            "videoId": video_id,
            "textFormat": "plainText",
            "key": api_key
        }
        if next_page_token:
            params["pageToken"] = next_page_token
        comments_response = requests.get(url, params=params)
        response_json = comments_response.json()
        items = response_json.get("items", [])
        comments += [item["snippet"]["topLevelComment"]["snippet"]["textOriginal"] for item in items]
        next_page_token = response_json.get("nextPageToken")
        if not next_page_token:
            break
    return comments


# Load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

# GCP API key
api_key = "..."
video_id = "..."
allComments = fetch_all_comments(video_id, api_key)
print(allComments[:15])

input_texts = ["Hello, my dog is cute", "I hate this movie", "This is the best day of my life"]
sentiments = []

for input_text in input_texts:
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

    sentiments.append(sentiment)

print(sentiments)
