import pandas as pd

def preprocess_data(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)
    
    # Example preprocessing steps
    data.fillna(method='ffill', inplace=True)
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)
    
    return data

if __name__ == "__main__":
    data = preprocess_data('data/weather_data.csv')
    data.to_csv('data/preprocessed_weather_data.csv')
