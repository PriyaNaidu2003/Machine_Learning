Breast Cancer Detection Project

This project aims to develop a web application that predicts whether a breast cancer tumor is benign or malignant based on various features. The system uses a Logistic Regression model trained on a breast cancer dataset and allows users to input specific tumor-related features through a web interface to receive predictions.

Key Features:


Model: Logistic Regression for binary classification.

Dataset: Used a publicly available breast cancer dataset with features such as radius, texture, and area.

Technologies:


Python, Flask for the web framework.

Pandas for data manipulation.

scikit-learn for model training and evaluation.

gdown for downloading the dataset from Google Drive.

pickle for saving and loading the trained model and scaler.


Prediction Interface: A simple HTML form allows users to enter the tumor's characteristics, and the model returns a prediction of either "Benign" or "Malignant."


How it Works:


The dataset is downloaded from Google Drive and preprocessed.

The Logistic Regression model is trained on the dataset.

A web application built using Flask allows users to input the tumor's feature values.

The model predicts whether the tumor is malignant or benign based on the entered values.
