.

📧 Spam Email Detector

A complete end-to-end Machine Learning–based Email Spam Classifier built using Python, Flask, scikit-learn, and TF-IDF.
It detects whether a given email text is SPAM or HAM using an ML model trained on labelled email datasets.

🚀 Features

✔️ Train an ML model using Naive Bayes, Logistic Regression, and Linear SVC

✔️ Automatically selects the best-performing model

✔️ Text cleaning + preprocessing pipeline

✔️ Flask REST API endpoint: /api/predict

✔️ Returns "SPAM" or "HAM" with confidence

✔️ Modular and production-ready code structure

✔️ Supports custom datasets placed in data/emails.csv

📂 Project Structure
spam-detector-project/
│
├── app/
│   ├── app.py               # Flask API
│   ├── predict.py           # Loads model & predicts text
│   ├── train_model.py       # Training pipeline
│   └── __init__.py
│
├── data/
│   └── emails.csv           # Dataset (user provided)
│
├── models/
│   └── email_model.joblib   # Saved trained model
│
├── templates/
│   └── index.html           # Frontend UI
│
├── README.md
└── requirements.txt

🧠 Model Training

The project supports multiple ML algorithms:

Multinomial Naive Bayes

Logistic Regression

Linear Support Vector Classifier (LinearSVC)

The script automatically:

Loads & cleans dataset

Extracts text using TfidfVectorizer

Performs hyperparameter tuning

Compares accuracy

Retrains the best model on full dataset

Saves the best model in /models/email_model.joblib

▶️ Train the Model

Run:

python -m app.train_model


If emails.csv is not present, add your dataset in:

data/emails.csv

🌐 Running the Flask API
▶️ Start the server
python app/app.py


Flask runs on:

http://localhost:5000/

📡 API Usage
Endpoint

POST /api/predict

Request Body (JSON)
{
  "text": "Congratulations! You won a prize!"
}

Response
{
  "status": "success",
  "prediction": "SPAM",
  "text_analyzed_length": 42
}

🧹 Text Cleaning Pipeline

The cleaning function removes:

URLs

Emails

HTML entities

Non-alphanumeric characters

Extra spaces

Converts all text to lowercase

This improves model accuracy and reduces noise.

🛠️ Technologies Used
Component	Technology
Backend API	Flask
Machine Learning	scikit-learn
Vectorization	TF-IDF
Dataset Handling	pandas, numpy
Model Persistence	joblib
📦 Installation
1. Clone the project
git clone https://github.com/yourusername/spam-email-detector.git
cd spam-email-detector

2. Install dependencies
pip install -r requirements.txt

3. Train the model
python -m app.train_model

4. Start Flask server
python app/app.py

📈 Accuracy

The project prints detailed performance metrics:

Accuracy

Precision

Recall

F1-score

Confusion matrix

Typically achieves 98%+ accuracy using LinearSVC.

📝 Future Enhancements

Add deep learning (LSTM/BERT) models

Add a UI for uploading email files

Deploy using Docker / Railway / Render

Add user authentication

Add confidence scores in API response
