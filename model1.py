import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Download historical stock data
ticker = 'AAPL'  # You can change to any stock like 'GOOG', 'TSLA', etc.
df = yf.download(ticker, start='2015-01-01', end='2023-12-31')

# Preprocess Data
df = df[['Close']].copy()
df['Prediction'] = df[['Close']].shift(-30)  # Predict the closing price 30 days into the future

# Features (X) and labels (y)
X = np.array(df.drop(['Prediction'], axis=1))[:-30]
y = np.array(df['Prediction'])[:-30]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluation
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.4f}")

# Visualize Predictions
plt.figure(figsize=(12,6))
plt.plot(y_test, label='Actual Price', color='blue')
plt.plot(predictions, label='Predicted Price', color='red')
plt.title(f'{ticker} Stock Price Prediction')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
plt.show()

# Root Mean Squared Error (RMSE): 15.23
# R² Score: 0.3343
