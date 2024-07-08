import pandas as pd
from transformers import pipeline

def summarize_weather_trends(text):
    summarizer = pipeline('summarization')
    summary = summarizer(text, max_length=50, min_length=25, do_sample=False)
    return summary[0]['summary_text']

if __name__ == "__main__":
    data = pd.read_csv('data/preprocessed_weather_data.csv')
    text_data = data['temperature'].astype(str).tolist()
    text = ' '.join(text_data)
    summary = summarize_weather_trends(text)
    print("Weather Trends Summary:")
    print(summary)
