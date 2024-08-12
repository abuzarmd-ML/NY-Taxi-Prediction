import pandas as pd

def preprocess(df):
    # Rename columns
    df.rename(columns={
        'tpep_pickup_datetime': 'request_datetime', 
        'Airport_fee': 'Airport'
    }, inplace=True)

    df['request_datetime'] = pd.to_datetime(df['request_datetime'], errors='coerce')
    if 'tpep_dropoff_datetime' in df.columns:
        df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'], errors='coerce')
        df = df.dropna(subset=['tpep_dropoff_datetime'])
        df['duration_sec'] = (df['tpep_dropoff_datetime'] - df['request_datetime']).dt.total_seconds()
        df = df[(df['trip_distance'] > 0) & (df['duration_sec'] > 0)]
    else:
        df['duration_sec'] = 0  
        df = df[(df['trip_distance'] > 0)] 

    df['Airport'] = (df['Airport'] > 0).astype(int)

    df['request_datetime'] = df['request_datetime'].dt.hour

    feature_columns = [
        'request_datetime',
        'trip_distance',
        'PULocationID',
        'DOLocationID',
        'Airport'
    ]
    X = df[feature_columns]
    y = df['duration_sec']

    return X, y
