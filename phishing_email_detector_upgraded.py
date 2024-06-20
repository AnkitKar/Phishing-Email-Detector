import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import joblib
import tkinter as tk
from tkinter import messagebox, scrolledtext

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Load the dataset
data = pd.read_csv('enhanced_email_dataset.csv')


# Text preprocessing function
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'\W+', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stopwords.words('english')])
    return text


# Apply preprocessing
data['text'] = data['text'].apply(preprocess_text)

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['text'])

# Labels
y = data['label']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Save the model and vectorizer
joblib.dump(model, 'phishing_email_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

# Predict on the test set
y_pred = model.predict(X_test)

# Print accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Print confusion matrix
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Print classification report
print("Classification Report:\n", classification_report(y_test, y_pred))


# Function to predict if an email is phishing or legitimate
def predict_email(email):
    email = preprocess_text(email)
    email_vec = vectorizer.transform([email])
    prediction = model.predict(email_vec)
    return prediction[0]


# Test the model with a new email
new_email = "Dear user, please update your payment information immediately to avoid account suspension."
print("Prediction:", predict_email(new_email))


# GUI Application
def on_predict():
    email_text = email_entry.get("1.0", tk.END).strip()
    if not email_text:
        messagebox.showwarning("Input Error", "Please enter the email text.")
        return

    prediction = predict_email(email_text)
    result_label.config(text=f"Prediction: {prediction}")


# Create the main window
window = tk.Tk()
window.title("Phishing Email Detector")

# Create a text entry widget for the email text
tk.Label(window, text="Enter Email Text:").pack(pady=5)
email_entry = scrolledtext.ScrolledText(window, width=60, height=15)
email_entry.pack(pady=5)

# Create a button to predict
predict_button = tk.Button(window, text="Predict", command=on_predict)
predict_button.pack(pady=10)

# Create a label to display the prediction result
result_label = tk.Label(window, text="Prediction: ")
result_label.pack(pady=5)

# Start the GUI event loop
window.mainloop()
