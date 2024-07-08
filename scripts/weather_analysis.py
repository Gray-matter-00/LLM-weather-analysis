import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

def analyze_weather(data, model):
    future = model.make_future_dataframe(periods=365)
    forecast = model.predict(future)
    
    fig = model.plot(forecast)
    plt.title('Weather Forecast for New York')
    plt.xlabel('Date')
    plt.ylabel('Temperature')
    plt.show()

if __name__ == "__main__":
    data = pd.read_csv('data/preprocessed_weather_data.csv')
    model = Prophet()
    model = model.load('models/prophet_model.pkl')
    analyze_weather(data, model)
