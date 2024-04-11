# Project 6- Pharmaceutical Sale Prediction 
 Sale Prediction - Rossman Pharma

# Project Overview
Project Name: Sales Forecasting Enhancement for Rossmann Pharmaceuticals

Objective: Develop and implement an advanced sales forecasting system.

Key Features: Leveraging machine learning techniques, incorporating factors such as promotions, competition, holidays, 

seasonality, and locality.

Accurately predict sales across all stores, six weeks in advance, to support informed decision-making by the finance team.

# Introduction to the Rossmann Pharmaceuticals
Rossmann Pharmaceuticals: A Leading Name in Pharmaceutical Retail

Established Reputation: Trusted for Quality Healthcare Solutions

Nationwide Presence: Operating Across Multiple Cities and Regions

Commitment to Excellence: Providing Accessible Healthcare Products and Services

Strategic Focus: Embracing Innovation to Enhance Customer Experience and Operational Efficiency

Introduction to Sales Forecasting Project: Addressing Challenges and Opportunities for Growth

# Exploratory data analysis guided of hypotheses
Stores with a larger assortment should sell more

Stores with closer competitors should sell less

Stores with longer-standing competitors should sell more

Stores where products cost less for longer (active promotions) should sell more

Stores with more promotion days should sell more

Stores with more extended promotions should sell more

Stores open during Christmas holiday should sell more

Stores should sell more over the years

Stores should sell more in the second half of the year

Stores should sell more after the 10th day of each month

Stores should sell less on weekends

Stores should sell less during school holidays

Stores that open on Sundays should sell more

# Task 1: Exploration of Customer Purchasing Behavior
The initial phase involves a thorough exploration of customer behavior within various stores. Key aspects of this exploration include:

Cleaning and preprocessing the data to handle outliers and missing values.

Analyzing the impact of promotions and store openings on purchasing behavior.

Investigating seasonal purchase patterns, such as those during holidays.

Exploring correlations between sales, number of customers, and other factors.

Assessing the effectiveness of promotions and suggesting improvements.

Studying customer behavior concerning store opening and closing times.

Analyzing the influence of assortment type and competitor distance on sales.

The findings from this task will provide insights into customer behavior and help in understanding the factors affecting sales.

# Task 2: Prediction of Store Sales

In this phase, the project focuses on predicting daily store sales up to six weeks in advance. The process involves several subtasks:

Preprocessing the data by converting non-numeric columns, handling NaN values, and generating new features.

Scaling the data for better prediction accuracy.

Building machine learning models, initially focusing on tree-based algorithms like Random Forest Regressor.

Choosing and defending an appropriate loss function for sales prediction.

Conducting post-prediction analysis to explore feature importance and estimate confidence intervals.

Serializing the models with timestamps for daily predictions.

Employing deep learning techniques, specifically ARIMA, SARIMA, PROPHET, LSTM networks, to predict future sales.

# Deep Learning Model for Sales Prediction

This task involves building a LSTM regression model to predict future sales using time series data. The steps include:

Isolating the Rossmann Store Sales dataset into time series data.

Checking for stationarity and transforming the data accordingly.

Analyzing autocorrelation and partial autocorrelation.

Converting the time series data into supervised learning data.

Scaling the data and building the LSTM regression model.

# Task 3: Serving Predictions on a Web Interface

In this task, we implemented a Flask backend to serve predictions using our trained models and input parameters collected through a frontend interface. The objective is to provide an intuitive dashboard for store managers to input required parameters and receive predictions for sales amount and customer numbers.

# Rossmann Sales Prediction PPT

# Overall Analysis, Interpretation & Conclusion 







