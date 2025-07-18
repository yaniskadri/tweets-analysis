from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("data354/camembert-fr-covid-tweet-sentiment-classification")
model = AutoModelForSequenceClassification.from_pretrained("data354/camembert-fr-covid-tweet-sentiment-classification")
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

app = FastAPI()

class Tweet(BaseModel):
    text: str   

@app.post("/analyze") #decorator (function enhancer) to define the endpoint, when a POST request is made to /analyze the below function will be called
def analyze_tweet(tweet: Tweet):
    result = classifier(tweet.text)
    return result

