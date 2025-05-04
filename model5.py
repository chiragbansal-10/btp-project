import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import talib

# 1. Load CSV
df = pd.read_csv("AAPL1424.csv", parse_dates=["Date"])
df.sort_values("Date", inplace=True)

# 2. Add technical indicators
df['SMA_20'] = df['Close'].rolling(window=20).mean()
df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
df['MACD'], _, _ = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)

df.dropna(inplace=True)

# 3. Define features and target
features = ['Open', 'High', 'Low', 'Volume', 'SMA_20', 'EMA_20', 'RSI', 'MACD']
X = df[features]
y = df['Close']

# 4. Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

# 5. Train XGBoost model
model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05)
model.fit(X_train, y_train)

# 6. Predict and evaluate
predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"MSE: {mse:.4f}, RMSE: {np.sqrt(mse):.4f}, R²: {r2:.4f}")

# 7. Plot actual vs predicted
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual')
plt.plot(predictions, label='Predicted')
plt.legend()
plt.title("XGBoost Stock Price Prediction")
plt.xlabel("Time")
plt.ylabel("Close Price")
plt.grid(True)
plt.tight_layout()
plt.show()

# on 1st dataset = AI.csv
# MSE: 4.8872, RMSE: 2.2107, R²: 0.8325

# on 2nd dataset = GOOG.csv
# MSE: 7.4947, RMSE: 2.7376, R²: 0.9813

# on 3rd dataset = AAPL1424.csv
# MSE: 46.2402, RMSE: 6.8000, R²: 0.8660
