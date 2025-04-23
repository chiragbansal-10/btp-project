import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
from tensorflow.keras.models import Model
import talib

# 1. Load and prepare CSV data
def load_data_from_csv(filepath):
    df = pd.read_csv(filepath, parse_dates=['Date'])
    df.sort_values('Date', inplace=True)
    df.set_index('Date', inplace=True)
    return df

# 2. Add technical indicators
def add_technical_indicators(df):
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
    df['MACD'], _, _ = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df.dropna(inplace=True)
    return df

# 3. Preprocess data for Transformers
def preprocess_data(df, time_step=60):
    features = ['Close', 'SMA_20', 'EMA_20', 'RSI', 'MACD']
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df[features].values)

    X, y = [], []
    for i in range(time_step, len(data_scaled)):
        X.append(data_scaled[i-time_step:i])
        y.append(data_scaled[i, 0])  # Predicting the close price

    return np.array(X), np.array(y), scaler

# 4. Build Transformer model
def build_transformer_model(input_shape):
    inputs = Input(shape=input_shape)
    x = LayerNormalization()(inputs)
    x = MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    x = Dropout(0.1)(x)
    x = LayerNormalization()(x)
    x = Dense(64, activation='relu')(x)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1)(x)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

# 5. Train and evaluate model
def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, scaler):
    model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=1)

    predictions = model.predict(X_test)
    predictions_rescaled = scaler.inverse_transform(np.hstack((predictions, np.zeros((len(predictions), 4)))))
    y_test_rescaled = scaler.inverse_transform(np.hstack((y_test.reshape(-1,1), np.zeros((len(y_test), 4)))))

    mse = mean_squared_error(y_test_rescaled[:, 0], predictions_rescaled[:, 0])
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_rescaled[:, 0], predictions_rescaled[:, 0])

    print(f'MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}')

    plt.figure(figsize=(10, 6))
    plt.plot(y_test_rescaled[:, 0], label='Actual')
    plt.plot(predictions_rescaled[:, 0], label='Predicted')
    plt.title('Transformer Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

# 6. Pipeline execution
file_path = 'AI.csv'  # Replace with your path
df = load_data_from_csv(file_path)
df = add_technical_indicators(df)

X, y, scaler = preprocess_data(df)

# Train/Test split
train_size = int(len(X) * 0.8)
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# Build and run Transformer model
model = build_transformer_model((X_train.shape[1], X_train.shape[2]))
train_and_evaluate_model(model, X_train, y_train, X_test, y_test, scaler)

# MSE: 123.8838, RMSE: 11.1303, R²: -8.8309



## --------------------REMOVE THIS SHIT-----------------------