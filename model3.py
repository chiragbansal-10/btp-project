import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout
import talib

# 1. Load CSV Data
def load_data_from_csv(filepath):
    df = pd.read_csv(filepath, parse_dates=['Date'])
    df.sort_values('Date', inplace=True)
    df.set_index('Date', inplace=True)
    return df

# 2. Add Technical Indicators (SMA, EMA, RSI, MACD)
def add_technical_indicators(df):
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
    df['MACD'], df['MACD_signal'], _ = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    
    df.dropna(inplace=True)  # Drop any rows with NaN values (if any indicators couldn't be calculated)
    return df

# 
# def preprocess_data_multi_features(df, time_step=60):
#     features = ['Close', 'SMA_20', 'EMA_20', 'RSI', 'MACD']
#     df_features = df[features].values
    
#     scaler = MinMaxScaler()
#     data_scaled = scaler.fit_transform(df_features)
    
#     X, y = [], []
#     for i in range(time_step, len(data_scaled)):
#         X.append(data_scaled[i-time_step:i])  # using past 60 time steps
#         y.append(data_scaled[i, 0])  # target is Close price

#     return np.array(X), np.array(y), scaler

# 3. Preprocess Data with Multiple Features
def preprocess_data_multi_features(df, time_step=60):
    features = ['Close', 'SMA_20', 'EMA_20', 'RSI', 'MACD']
    df_features = df[features].values

    # Separate scalers
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    # Fit the full feature scaler
    data_scaled = feature_scaler.fit_transform(df_features)

    # Fit target scaler on just the Close column (first column)
    close_prices = df[['Close']].values
    target_scaler.fit(close_prices)

    X, y = [], []
    for i in range(time_step, len(data_scaled)):
        X.append(data_scaled[i-time_step:i])          # past 60 steps of 5 features
        y.append(close_prices[i])                     # unscaled target for now

    y_scaled = target_scaler.transform(np.array(y))   # scale target after collecting

    return np.array(X), y_scaled, feature_scaler, target_scaler


# 4. Build GRU Model
def build_gru_model(input_shape):
    model = Sequential()
    model.add(GRU(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(GRU(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

#
# def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, scaler):
#     model.fit(X_train, y_train, epochs=25, batch_size=32, verbose=1)
#     predictions = model.predict(X_test)
#     predictions_rescaled = scaler.inverse_transform(predictions)
#     y_test_rescaled = scaler.inverse_transform(y_test)

#     mse = mean_squared_error(y_test_rescaled, predictions_rescaled)
#     rmse = np.sqrt(mse)
#     r2 = r2_score(y_test_rescaled, predictions_rescaled)

#     print(f'MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}')

#     # Plot the results
#     plt.figure(figsize=(10, 6))
#     plt.plot(y_test_rescaled, label='Actual')
#     plt.plot(predictions_rescaled, label='Predicted')
#     plt.title('Stock Price Prediction')
#     plt.xlabel('Time')
#     plt.ylabel('Price')
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

# 5. Train & Evaluate the Model
def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, target_scaler):
    model.fit(X_train, y_train, epochs=25, batch_size=32, verbose=1)

    predictions = model.predict(X_test)

    predictions_rescaled = target_scaler.inverse_transform(predictions)
    y_test_rescaled = target_scaler.inverse_transform(y_test)

    mse = mean_squared_error(y_test_rescaled, predictions_rescaled)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_rescaled, predictions_rescaled)

    print(f'MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}')

    plt.figure(figsize=(10, 6))
    plt.plot(y_test_rescaled, label='Actual')
    plt.plot(predictions_rescaled, label='Predicted')
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



# 6. Running the Full Pipeline
file_path = 'AI.csv'  # Replace with your actual path
df = load_data_from_csv(file_path)
df = add_technical_indicators(df)  # Add technical indicators

# X, y, scaler = preprocess_data_multi_features(df)
X, y, feature_scaler, target_scaler = preprocess_data_multi_features(df)

# Train/Test Split
train_size = int(len(X) * 0.8)
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# Choose model: LSTM or GRU
model = build_gru_model((X_train.shape[1], X_train.shape[2]))

# Train and evaluate the model
train_and_evaluate_model(model, X_train, y_train, X_test, y_test, target_scaler)

# MSE: 9.8853, RMSE: 3.1441, R²: 0.2155

## --------------------REMOVE THIS SHIT-----------------------