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

<!-- ========== INTRODUCTION ====================== -->

In quantitative finance, forecasting stock prices is one of
the most intriguing yet difficult challenges. Stock markets
are governed by a variety of factors—macroeconomic indi-
cators, investor emotions, international news, and others—so
that price fluctuations become inherently nonlinear, dynamic,
and even unpredictable. Conventional statistical methods, as
important as they are, are often not as flexible to deal with
such volatility. To tackle this, Artificial Intelligence (AI) has
proved to be a strong ally, which can dig deep insights from
masses of past stock data.
This project ”Stock Analysis using AI” aims to utilize
three different modeling techniques—Linear Regression, Long
Short-Term Memory (LSTM) networks, and XGBoost Re-
gression—to forecast stock prices based on an assortment
of datasets (AAPL, GOOG, AI). Each model is selected in
order to depict a unique category of learning: statistical, deep
learning, and ensemble-based machine learning. Comparative
study enables us to learn how every model accommodates the
financial time-series space.

<!-- Ye cheez models ke liye hai -->
In this research, three different predictive modeling techniques were investigated and contrasted to predict stock closing prices: Linear Regression, LSTM, and XGBoost. The Linear Regression model, which is a baseline, is a simple statistical technique that tries to predict the closing price of a stock 30 days ahead based only on its current price. Its main strengths are its simplicity, interpretability, and computational cost. But it inherently makes a linear assumption in the data and does not take into account technical indicators or temporal dependencies. Tested against the AAPL dataset, Linear Regression produced a Root Mean Squared Error (RMSE) of 15.23 and a value of 0.3343 for R², reflecting poor predictive capacity.

To overcome the limitations of the linear models, an LSTM network was implemented, taking advantage of its capability to capture temporal sequences and long-term dependencies. Here, the model utilized 60 days' worth of data on closing prices to forecast the price on the next day. The data was scaled with MinMaxScaler, and the model structure comprised two stacked LSTM layers with dropout regularization followed by a dense output layer. The LSTM model showed a significant improvement in performance on all datasets. In particular, it attained an RMSE of 3.6687 (R² = 0.4083) on the AI dataset, 7.1526 (R² = 0.8734) on the GOOG dataset, and 6.1630 (R² = 0.8910) on the AAPL dataset. These findings serve to emphasize the model's resilience in detecting embedded trends and expressing nonlinear temporal tendencies, particularly over large and well-established datasets. 

The third method employed XGBoost (Extreme Gradient Boosting), which is a versatile ensemble-based methodology renowned for tackling structured tabular data. This model combined both general stock features (Open, High, Low, Volume) and technical features specific to the domain like Simple Moving Average (SMA), Exponential Moving Average (EMA), Relative Strength Index (RSI), and Moving Average Convergence Divergence (MACD). XGBoost addressed the nonlinearity in stock data quite well and gained from feature engineering and regularization techniques. It performed better than the baseline Linear Regression model and matched LSTM in many instances. On the AI dataset, it reported an RMSE of 2.2107 (R² = 0.8325); on the GOOG dataset, 2.7376 (R² = 0.9813); and on the AAPL dataset, 6.8000 (R² = 0.8660). The results reflect the strength and accuracy of XGBoost, particularly when combined with technical indicators, and are therefore a viable contender for stock price prediction.

<!--  What is the motivation for using these models -->
The motivation for using AI models in stock price prediction stems from the limitations of traditional statistical methods, which assume linearity and fail to capture the complex, nonlinear, and time-dependent patterns in financial data. Stock prices are influenced by a range of dynamic factors, including historical trends, technical indicators, and market behavior. AI models like LSTM can learn from sequential patterns, while XGBoost effectively leverages engineered features. These models offer improved accuracy and adaptability, making them better suited for forecasting in volatile market conditions. Thus, AI provides a powerful approach to enhance prediction reliability in financial analytics

<!-- data description -->
In this research, we used three varied stock market datasets to analyze and compare the performance of various predictive models. The data was obtained from reliable financial websites—Yahoo Finance (using the yfinance Python package) and Kaggle—to guarantee reliability and current financial precision. We chose three firms that exhibit different types of market behaviors: AAPL (Apple Inc.), a large and stable technology firm with great trading volume and market stability; GOOG (Alphabet Inc.), another tech giant with considerable historical data and liquidity; and AI (C3.ai Inc.), a comparatively new and highly volatile AI software firm. These choices allowed us to evaluate model robustness with varying levels of volatility, stability, and market maturity.

All datasets included fundamental financial characteristics like Open, High, Low, Close prices, and Volume, which are imperative for time-series forecasting and technical analysis. The duration of coverage ranged from January 2015 to December 2023, providing close to nine years of daily trading data and more than 2,000 records per stock, providing sufficient data for training and testing of models, especially deep learning models.

To pre-process the datasets for model training, several preprocessing operations were used. The missing values—the usually encountered entries in the early or last slots—were tackled through forward-filling methods or eliminated depending on data adequacy. Feature scaling was performed employing the MinMaxScaler to adjust the data on a [0, 1] scale range, which proves particularly useful to models such as LSTM by strengthening numerical stability as well as convergence. The target variable was defined specifically for each model: for Linear Regression, it was defined as the closing price 30 days later (with a shift(-30) transformation); for LSTM, a sliding window approach was taken with the closing prices of the last 60 days to forecast the following day; and for XGBoost, the next-day actual Close price was utilized along with engineered technical features like SMA, EMA, RSI, and MACD for improving prediction ability. These preprocessing choices ensured consistency across models and improved the learning process by standardizing inputs and incorporating temporal and domain-specific insights.

<!-- LITERATURE REVIEW -->
<!-- METHODOLOGY -->
To compare different predictive models in stock price prediction, we used three different models: Linear Regression, LSTM (Long Short-Term Memory), and XGBoost Regressor, each of which is a different predictive algorithm.

5.1 Linear Regression:
Linear Regression was our control model because it is easy to interpret and understand. It makes a linear relationship between input and target features, so it is simple but restrictive in addressing the nonlinear and unstable nature of financial markets. In this project, the model was trained to forecast the closing price of the stock 30 days ahead with a shift(-30) operation. Although simple to deploy with minimal computational expense, its built-in assumptions overlook the sequential interdependence of stock prices and are therefore less appropriate for long-term predictions in changing conditions.

5.2 LSTM Model:
LSTM, a specific type of Recurrent Neural Networks (RNNs), was used to model temporal relationships and intricate patterns that form a part of stock market data. The model was constructed with two stacked LSTM and dropout regularization to avoid overfitting, followed by a dense output layer for making the final prediction. All of the training inputs were a sequence of 60 consecutive days' closing prices, so the model could pick up on prior context and learn from them. The data was normalized by MinMaxScaler prior to being fed into the network. Training was performed over several epochs with a constant batch size, aided by time-series data splits to maintain chronological order. The architecture of the memory cells in LSTM allowed it to access long-term dependencies and thus existed as a robust sequential forecasting tool.

5.3 XGBoost Regressor:
We applied the XGBoost gradient boosting decision tree algorithm because of strong performance on structured data and regularization techniques-based robustness against overfitting. In comparison with LSTM, learning sequential data, XGBoost had strong feature engineering. We supplemented the dataset by incorporating domain-specific technical factors such as Simple Moving Average (SMA), Exponential Moving Average (EMA), Relative Strength Index (RSI), and Moving Average Convergence Divergence (MACD). These features enabled the model to comprehend short-term and long-term price movements, momentum, and market volatility. The hyperparameters like learning rate, depth, and number of estimators were set in order to gain maximum performance on every set of data. XGBoost worked effectively with non-linear relations and gave consistent results on various stocks, rendering it a prominent contender together with LSTM.

<!-- EXPERIMENTAL SETUP -->
6. Experimental Setup
The experiments were run within a Python environment, using some strong libraries for machine learning, deep learning, and technical analysis. TensorFlow was used for the creation and training of the LSTM model, scikit-learn to create Linear Regression, data preprocessing tasks, and metrics for evaluation, and XGBoost to execute gradient boosting regression. Furthermore, TA-Lib was used to calculate domain-specific technical indicators like SMA, EMA, RSI, and MACD, which proved helpful in enriching feature richness for the XGBoost model. The code was run on a machine having an Intel Core i7 CPU (11th Gen), 16GB RAM, and NVIDIA RTX 3060 GPU (6GB VRAM), which offered adequate computing capacity for training deep learning models at an optimal pace.

To assess model performance, we employed conventional regression measures such as Root Mean Squared Error (RMSE), Mean Squared Error (MSE), and R² Score (Coefficient of Determination). These measures gave a thorough picture of the accuracy of prediction and model fit. Each stock's dataset was divided into training and testing sets based on a conventional 80:20 split, where 80% of the historical data were utilized for training and 20% for testing. In models such as time series models LSTMs, a rolling window strategy was taken to ensure temporal consistency. There was an experiment setup that used fair comparison amongst the models based on best financial time-series forecasting practises.

<!-- Results and Discussion -->
The accuracy of the three predictive models—Linear Regression, LSTM, and XGBoost—was evaluated with three sets of stock data: AAPL, GOOG, and AI. Detailed comparison was done with quantitative metrics as well as graphical analysis. Line graphs were plotted for each set to show the relation between the real and predicted stock prices to determine the accuracy of the prediction through visual inspection. Additionally, an abridged table of significant error measures such as RMSE, MSE, and R² Score was prepared to display a numerical basis for model performance analysis.

The results showed that Linear Regression, though simple and intuitive, performed poorly with non-linearity and temporal dependencies, returning lower R² scores (e.g., 0.3343 on AAPL). The LSTM model demonstrated to have good capability to learn time-series trends, especially on stable and large data like AAPL and GOOG, with the R² measures as high as 0.8910 and 0.8734, respectively. Nonetheless, it did relatively lower on the more volatile AI dataset. The XGBoost model was a strong contender and performed well across all datasets, particularly GOOG (R² = 0.9813) and AI (R² = 0.8325), due to the incorporation of technical indicators like SMA, EMA, RSI, and MACD.

Utilization of technical indicators clearly enhanced the predictability of the XGBoost model, most notably in terms of momentum detection and trend reversal. While LSTM learned better long-term trends, it was more prone to overfitting with small datasets without proper regularization. In contrast, XGBoost demonstrated better generalizability, which might be due to the ensemble architecture and strong regularization used. Overall, the results show that there is no one-size-fits-all model; rather, performance varies based on the data profile, with XGBoost being superior on feature-rich structured data and LSTM more appropriate for smoother, temporally consistent datasets.

<!-- CONCLUSION -->
In our research, we investigated and contrasted the performance of three predictive models—Linear Regression, LSTM, and XGBoost—in stock price forecasting based on past data. Through our research, we identify that although Linear Regression offers a straightforward and understandable baseline model, it is incapable of being able to detect non-linear and sequential patterns in stock market data. Contrariwise, LSTM and XGBoost greatly exceeded Linear Regression's performance on every dataset, and while XGBoost made use of engineered technical features, LSTM effectively captured temporal dynamics.

What this indicates most of all is that models capable of including time-series dynamics (LSTM) or feature-fectched within domains (XGBoost) produce dramatically greater levels of prediction accuracy. The lesson learned from this is both the need for temporal modeling as well as richer features in terms of stock price prediction. But our method has its limitations. The models make use of only technical indicators and do not factor in external data like real-time financial news, economic announcements, or sentiment, which can affect stock prices too. Future efforts can incorporate these elements to make predictions even stronger.

<!-- Future Prospect -->
Although this study has achieved encouraging results by using LSTM and XGBoost models, there are significant avenues of further improvement. One major extension would be to integrate real-time financial news and social media sentiment analysis by employing Natural Language Processing (NLP) methods because such extraneous factors tend to influence short-term market movements. Moreover, the use of more sophisticated deep learning models like Transformer-based time series models (e.g., Temporal Fusion Transformers or Informers) might be able to learn intricate dependencies over longer periods.

Subsequent research may also widen the scope by extending to heterogeneous asset classes like cryptocurrencies, ETFs, and commodities, thus putting the model through its paces in a variety of financial landscapes. Lastly, creating hybrid ensemble models that synergize the respective strengths of LSTM (sequential learning) and XGBoost (feature-based decision making) might yield better generalization and precision. Such optimizations can have a substantial impact on the practical applicability of AI-based models to real-world financial forecasting.
