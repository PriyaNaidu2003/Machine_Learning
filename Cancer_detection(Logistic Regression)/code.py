import pandas as pd
import gdown

import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def download_file(file_id, file_path):
    download_url = f'https://drive.google.com/uc?id={file_id}'
    try:
        # Attempt to download the file
        gdown.download(download_url, file_path, quiet=False)
        print("File downloaded successfully!")
    except Exception as e:
        print(f"Error downloading file: {e}")
        raise

# Downloading the CSV file from Google Drive
file_id = '1mGyb3Z4aD02fNkyxUF0q8uL_j5XJR2Hv'
file_path = 'data (1).csv'

download_file(file_id, file_path)

# Loading the dataset into a pandas DataFrame
df = pd.read_csv(file_path)
# Print the columns to check if the names match


# Drop columns with missing values
df = df.dropna(axis=1)

# Encode the 'diagnosis' column
lb = LabelEncoder()
df['diagnosis'] = lb.fit_transform(df['diagnosis'])

# Select specific features
features = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean',
    'fractal_dimension_mean'
]

# Check if the specified features exist in the DataFrame
missing_features = [feature for feature in features if feature not in df.columns]
if missing_features:
    print(f"Missing features in the dataset: {missing_features}")
else:
    X = df[features].values
    y = df['diagnosis'].values

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Standardize the features
    st = StandardScaler()
    X_train = st.fit_transform(X_train)
    X_test = st.transform(X_test)

    # Train the logistic regression model
    l = LogisticRegression()
    l.fit(X_train, y_train)

    # Evaluate the model
    train_score = l.score(X_train, y_train)
    test_score = accuracy_score(y_test, l.predict(X_test))
    report = classification_report(y_test, l.predict(X_test))

    print(f"Training Accuracy: {train_score}")
    print(f"Testing Accuracy: {test_score}")
    print(f"Classification Report:\n{report}")

    # Save the model and scaler
    with open('model.pkl', 'wb') as model_file:
        pickle.dump(l, model_file)

    with open('scaler.pkl', 'wb') as scaler_file:
        pickle.dump(st, scaler_file)


# Save the model and scaler
with open('model.pkl', 'wb') as model_file:
    pickle.dump(l, model_file)

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(st, scaler_file)


# Save the model and scaler
with open('model.pkl', 'wb') as model_file:
    pickle.dump(l, model_file)

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(st, scaler_file)

from flask import Flask, render_template_string, request

import pickle
import numpy as np

# Load the trained model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def home():
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Cancer Detection</title>
    </head>
    <body>
        <h2>Breast Cancer Detection</h2>
        <h3>Enter Features</h3>
        <form action="/predict" method="post">
            <input type="text" name="radius_mean" placeholder="Radius Mean"><br>
            <input type="text" name="texture_mean" placeholder="Texture Mean"><br>
            <input type="text" name="perimeter_mean" placeholder="Perimeter Mean"><br>
            <input type="text" name="area_mean" placeholder="Area Mean"><br>
            <input type="text" name="smoothness_mean" placeholder="Smoothness Mean"><br>
            <input type="text" name="compactness_mean" placeholder="Compactness Mean"><br>
            <input type="text" name="concavity_mean" placeholder="Concavity Mean"><br>
            <input type="text" name="concave_points_mean" placeholder="Concave Points Mean"><br>
            <input type="text" name="symmetry_mean" placeholder="Symmetry Mean"><br>
            <input type="text" name="fractal_dimension_mean" placeholder="Fractal Dimension Mean"><br>
            <input type="submit" value="Predict">
        </form>
    </body>
    </html>
    ''')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        features = []
        for feature_name in [
            'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
            'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean',
            'fractal_dimension_mean'
        ]:
            value = request.form.get(feature_name)
            if value is None or value.strip() == '':
                return render_template_string('''
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Prediction Error</title>
                </head>
                <body>
                    <h2>Error</h2>
                    <p>The field "{{ feature_name }}" is required and cannot be empty.</p>
                    <a href="/">Go Back</a>
                </body>
                </html>
                ''', feature_name=feature_name)
            try:
                features.append(float(value))
            except ValueError:
                return render_template_string('''
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Prediction Error</title>
                </head>
                <body>
                    <h2>Error</h2>
                    <p>The field "{{ feature_name }}" must be a valid number.</p>
                    <a href="/">Go Back</a>
                </body>
                </html>
                ''', feature_name=feature_name)

        final_features = [np.array(features)]

        # Scale the input features
        scaled_features = scaler.transform(final_features)

        # Make prediction
        prediction = model.predict(scaled_features)
        if prediction == 1:
            prediction = 'Malignant'
        elif prediction == 0:
            prediction = 'Benign'

        return render_template_string('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Prediction Result</title>
        </head>
        <body>
            <h2>Prediction Result</h2>
            <p>The prediction is: {{ prediction }}</p>
            <a href="/">Go Back</a>
        </body>
        </html>
        ''', prediction=prediction)
    except Exception as e:
        return str(e)


if __name__ == '__main__':
    app.run(debug=True)
