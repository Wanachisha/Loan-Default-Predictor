Data Science Project: Credit Risk Analysis and Loan Default Prediction

This project aims to enhance my knowledge of core data science concepts such as data preprocessing, exploratory data analysis, and model fitting. I will conduct a credit risk analysis and build a loan default prediction system using parameters derived from a real-world dataset.

Project Description

This project analyses a real-world credit dataset to build a loan default detection system. The analysis establishes data-driven parameters defining what constitutes loan default and loan delinquency, which then informs the design of two predictive classification models: Logistic Regression and a Linear Support Vector Machine (SVM).

The full pipeline follows the sequence: data cleaning -> parameter definition -> encoding -> scaling -> model training -> evaluation.

About the Dataset

Source: Kaggle


Project Steps

1. Preprocessing Tasks

- Handled missing or inconsistent data
- Removed extreme outliers
- One-hot encoded categorical variables to convert them into a numerical format
- Split the dataset into feature (X) and target (y) variables

2. Model Selection, Fitting, Evaluation, and Testing

- Built a Pipeline for Logistic Regression and Linear SVM.
  - The pipeline includes: imputation, encoding, scaling, and algorithm training.
- Evaluated the performance of both models against the test set.
- Provided insights into the effectiveness of the two models and determined which is best suited for the system.

3. Conclusion

- Summarised key findings from the exploratory analysis and model testing.


A Streamlit dashboard was created to test the parameters set against data outside the datset.

Run the application:

> python -m streamlit run app.py

The application will automatically open in your default web browser at http://localhost:8501.



Author

_Wana Chisha Munjunga_

_Data Science Portfolio Project_
