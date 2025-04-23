ABSTRACT

In this study, we examine the use of artificial intelligence in predicting stock market trends by developing a forecast model that integrates previous price information with popular technical indicators. We use the XGBoost regression algorithm, a high-performance machine learning methodology that is renowned for its speed and precision, to forecast stock closing prices. The data set provides metrics such as 20-day Simple and Exponential Moving Averages (SMA and EMA), Relative Strength Index (RSI), and MACD, which are typically employed by traders to analyze market trends. Through training and testing against actual market data, our model obtained an R² value of 0.8325, showing high predictive consistency. This outcome indicates that blending AI models and technical analysis is capable of drastically improving forecast precision, providing beneficial insights to both investors and analysts.

Keywords: Stock Prediction, XGBoost, Technical Analysis, RSI, EMA, SMA, MACD, Machine Learning, AI in Finance




Give the brief overview of the model which gives R2 as 0.8

Absolutely! Here's a brief overview of the XGBoost-based stock price prediction model that gave you an R² of 0.8:

📈 Model Overview: XGBoost Stock Price Predictor
🧠 Model Type
XGBoost Regressor (Extreme Gradient Boosting)
A highly efficient and accurate tree-based machine learning model optimized for tabular data and regression tasks.

📊 Data Used
You used historical stock data with columns:

Date, Open, High, Low, Close, Adj Close, Volume

You enhanced this data with technical indicators using ta-lib:

SMA_20 – 20-day Simple Moving Average

EMA_20 – 20-day Exponential Moving Average

RSI – Relative Strength Index (momentum)

MACD – Moving Average Convergence Divergence (trend strength)

🏗️ Features Used for Prediction
The following features were used as inputs to the model:

Open, High, Low, Volume

SMA_20, EMA_20, RSI, MACD

The target variable:

Close price (what the model is predicting)

🧪 Training & Evaluation
The dataset was split into training (80%) and testing (20%) without shuffling (to preserve time order).

XGBoost was trained using 200 trees (n_estimators=200) with a learning rate of 0.05.

Model performance was evaluated using:

MSE (Mean Squared Error)

RMSE (Root Mean Squared Error)

R² (Coefficient of Determination) = 0.8

📉 What Does R² = 0.8 Mean?
The model is able to explain 80% of the variance in the stock’s closing price.

It indicates strong predictive power for stock price movement based on historical data + technical indicators.


Component | Details
Model | XGBoostRegressor (Gradient Boosted Trees)
Input Features | OHLC + Volume + SMA, EMA, RSI, MACD
Target | Close Price
Accuracy | R² = 0.8
Tools Used | pandas, xgboost, sklearn, matplotlib, ta-lib
Forecast Type | Short-term regression (not classification)