import pandas as pd
from prophet import Prophet

def train_model(data):
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.fit(data)
    return model

if __name__ == "__main__":
    data = pd.read_csv('data/preprocessed_weather_data.csv')
    data.rename(columns={'date': 'ds', 'temperature': 'y'}, inplace=True)
    model = train_model(data)
    model.save('models/prophet_model.pkl')
