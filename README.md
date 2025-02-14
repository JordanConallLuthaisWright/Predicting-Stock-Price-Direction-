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
Clone this repository and open `classification.ipynb` in Jupyter Notebook:  

```bash
git clone https://github.com/your-username/predict-stock-direction.git
cd predict-stock-direction
jupyter notebook

## **Final Findings**  

The final model, an **SVC (Support Vector Classifier) with a linear kernel (C=100)**, was trained using a **StandardScaler** preprocessing step to normalize the feature data. The model was trained on the **combined train-test dataset** and evaluated on the **validation set**.  

### **Performance Metrics**  
- **Accuracy:** 80.9%  
- **AUC (Area Under ROC Curve):** 0.91  

### **Key Insights**  
- The model achieved a high **accuracy of 80.9%**, indicating strong predictive performance in classifying whether the **next day's closing price will be higher or lower than the opening price**.  
- The **AUC score of 0.91** suggests that the model is very effective at distinguishing between upward and downward price movements.  
- Using **SVM with a linear kernel** proved to be a **robust approach** for this classification problem, capturing relevant patterns in stock price data.  
- Feature scaling via **StandardScaler** played a crucial role in optimizing model performance.  

### **Limitations & Considerations**  
- The model is **only trained on historical price data** and does not incorporate external factors such as news sentiment, macroeconomic indicators, or fundamental data.  
- While the accuracy is high, stock market conditions change dynamically, meaning model performance may vary in real-world trading scenarios.  
- Further **hyperparameter tuning** or **feature engineering** (e.g., incorporating technical indicators) could improve model robustness and generalization.  

### **Next Steps**  
- Experimenting with **ensemble models** (e.g., **Random Forest, XGBoost**) to compare performance.  
- Exploring **deep learning models (e.g., LSTMs, GRUs)** for time-series forecasting.  
- Adding **technical indicators (RSI, MACD, Bollinger Bands)** to enhance feature set.  

## Contact
If you have any questions or would like further details about this project or my experience, feel free to reach out:
- **Email**: [jordan.c.l.wright@gmail.com](mailto:jordan.c.l.wright@gmail.com)


