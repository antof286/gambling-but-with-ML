import numpy as np
import pandas as pd
import yfinance as yf
import torch
from sklearn.preprocessing import StandardScaler

def fetch_and_prepare_data(ticker, start_date, end_date, seq_len, scaler=None):
    data = yf.download(ticker, start=start_date, end=end_date)[['Close', 'Volume']]
    data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))
    data = data.dropna()
    
    features = data[['Log_Return', 'Volume']].values
    
    if scaler is None:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(features)
    else:
        scaled_data = scaler.transform(features)
        
    X = []
    for i in range(len(scaled_data) - seq_len):
        X.append(scaled_data[i:i+seq_len])
        
    return torch.tensor(np.array(X), dtype=torch.float32), scaler, data
