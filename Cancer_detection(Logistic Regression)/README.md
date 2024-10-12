Breast Cancer Detection Project

This project aims to develop a web application that predicts whether a breast cancer tumor is benign or malignant based on various features. The system uses a Logistic Regression model trained on a breast cancer dataset and allows users to input specific tumor-related features through a web interface to receive predictions.

Key Features:


1.Model: Logistic Regression for binary classification.

2.Dataset: Used a publicly available breast cancer dataset with features such as radius, texture, and area.

Technologies:


1.Python, Flask for the web framework.

2.Pandas for data manipulation.

3.scikit-learn for model training and evaluation.

4.gdown for downloading the dataset from Google Drive.

5.pickle for saving and loading the trained model and scaler.


Prediction Interface: A simple HTML form allows users to enter the tumor's characteristics, and the model returns a prediction of either "Benign" or "Malignant."


How it Works:


1.The dataset is downloaded from Google Drive and preprocessed.

2.The Logistic Regression model is trained on the dataset.

3.A web application built using Flask allows users to input the tumor's feature values.

4.The model predicts whether the tumor is malignant or benign based on the entered values.
