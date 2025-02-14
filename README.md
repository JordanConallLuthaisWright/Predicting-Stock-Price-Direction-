# Predicting Stock Price Direction  

## Overview  
This repository contains my solution to a **StrataScratch** challenge, where I predict the daily price direction of **Amazon.com, Inc. (AMZN)** stock. The solution, implemented in `classification.ipynb`, demonstrates a **binary classification approach** to determine whether the **closing price** will be higher or lower than the **opening price** for the next trading day.  

Although the question and solution focus on AMZN, the methodology can be **generalized to other tickers** by pulling stock data via the `yfinance` API in Python.  

## Dataset  
The dataset consists of historical stock price data from **1997 to 2020**, which has been split into:  
- **Training Data:** 1997–2016 (`AMZN_train.csv`)  
- **Validation Data:** 2016–2018 (`AMZN_val.csv`)  
- **Testing Data:** 2018–2020 (`AMZN_test.csv`)  

Each dataset has the following columns:  

| Column    | Description                                      |
|-----------|--------------------------------------------------|
| `Date`    | Trading date (YYYY-MM-DD format)               |
| `Open`    | Stock price at market open                     |
| `High`    | Highest price of the day                       |
| `Low`     | Lowest price of the day                        |
| `Close`   | Stock price at market close                    |
| `Adj Close` | Adjusted closing price after corporate actions |
| `Volume`  | Number of shares traded                        |

## Objective  
The objective is to **train and evaluate a predictive model** that takes historical stock data as input and outputs a binary prediction:  
- **1 (Up)** → Closing price is higher than the opening price.  
- **0 (Down)** → Closing price is lower than the opening price.  

We focus on **classification metrics** rather than predicting exact stock prices, as **directional accuracy** is more relevant for trading decisions.  

## Approach  
1. **Data Preprocessing**  
   - Load data from CSV files  
   - Handle missing values and data formatting  
   - Feature engineering (e.g., moving averages, volatility measures)  

2. **Exploratory Data Analysis (EDA)**  
   - Visualizing stock trends and patterns  
   - Checking data distributions and correlations  

3. **Model Selection & Training**  
   - Implementing machine learning models (e.g., **Logistic Regression, Random Forest, XGBoost**)  
   - Tuning hyperparameters for optimal performance  
   - Evaluating models using **AUC (Area Under Curve)**, accuracy, and other classification metrics  

4. **Evaluation & Testing**  
   - Assessing model performance on validation and test data  
   - Interpreting feature importance  

## Key Insights  
- The stock market is highly volatile, making short-term predictions challenging.  
- Rather than predicting exact prices, **binary classification** models help identify trends.  
- A model achieving **AUC > 0.515** is considered sufficient for this challenge.  

## Practical Considerations  
- **No external data sources** were used beyond the provided dataset.  
- The focus is on code **structure, interpretability, and reproducibility** rather than maximizing model accuracy.  
- This project was designed to be completed within **3 hours**, balancing efficiency and depth of analysis.

## Dependencies  
To run this project, install the following Python libraries:  
```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost yfinance

## Running the Notebook 
Clone this repository and open classification.ipynb in Jupyter Notebook:
```bash
git clone https://github.com/your-username/predict-stock-direction.git
cd predict-stock-direction
jupyter notebook

## Future Improvements
- Testing additional feature engineering techniques
- Exploring deep learning models like LSTMs for sequential data
- Incorporating technical indicators to enhance predictive power



