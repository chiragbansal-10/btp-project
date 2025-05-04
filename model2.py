import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# 1. Load CSV Data
def load_data_from_csv(filepath):
    df = pd.read_csv(filepath, parse_dates=['Date'])
    df.sort_values('Date', inplace=True)
    df.set_index('Date', inplace=True)
    return df

# 2. Preprocessing (Using 'Close' prices)
def preprocess_data(df, column='Close', time_step=60):
    scaler = MinMaxScaler()
    data = df[[column]].values
    data_scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(time_step, len(data_scaled)):
        X.append(data_scaled[i-time_step:i])
        y.append(data_scaled[i])
    return np.array(X), np.array(y), scaler

# 3. Model Architecture
def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# 4. Training & Evaluation
def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, scaler):
    model.fit(X_train, y_train, epochs=25, batch_size=32, verbose=1)

    predictions = model.predict(X_test)
    predictions_rescaled = scaler.inverse_transform(predictions)
    y_test_rescaled = scaler.inverse_transform(y_test)

    mse = mean_squared_error(y_test_rescaled, predictions_rescaled)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_rescaled, predictions_rescaled)

    print(f'MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}')

    # Plot
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

# 🔧 Pipeline Execution
file_path = 'GOOG.csv'  # Replace with your actual path
df = load_data_from_csv(file_path)
X, y, scaler = preprocess_data(df, column='Close')
train_size = int(len(X) * 0.8)

X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

model = build_model((X_train.shape[1], 1))
train_and_evaluate_model(model, X_train, y_train, X_test, y_test, scaler)

# on 1st dataset = AI.csv
# MSE: 13.4596, RMSE: 3.6687, R²: 0.4083

# on 2nd dataset = GOOG.csv
# MSE: 51.1594, RMSE: 7.1526, R²: 0.8734

# on 3rd dataset = AAPL1424.csv
# MSE: 37.9827, RMSE: 6.1630, R²: 0.8910
