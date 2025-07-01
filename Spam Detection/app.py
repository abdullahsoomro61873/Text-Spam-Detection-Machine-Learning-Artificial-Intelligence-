import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy
from flask import Flask, request, render_template

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load the CSV file into a Pandas DataFrame
df = pd.read_csv(r'messages.csv')

# Preprocess data: tokenize, remove stopwords, and create a word frequency dictionary
def preprocess(message):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(message)
    words_filtered = [word.lower() for word in words if word.isalnum() and word.lower() not in stop_words]
    return {word: True for word in words_filtered}

# Create a new column for the preprocessed messages
df['preprocessed_message'] = df['message'].apply(preprocess)

# Create the dataset for the classifier
dataset = [(features, label) for label, features in zip(df['label'], df['preprocessed_message'])]

# Train the Naive Bayes classifier
train_data = dataset
classifier = NaiveBayesClassifier.train(train_data)

# Test the classifier interactively (this part is optional and for command-line testing)
# test_message = (input("Enter Text Here: "))
# print("Message:", test_message)
# print("Label:", classifier.classify(preprocess(test_message)))

# Check the accuracy of the classifier
print("Accuracy:", accuracy(classifier, train_data))

# Show the most informative features
classifier.show_most_informative_features()

# Create Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_message = request.form['message']
    processed_message = preprocess(user_message)
    classification = classifier.classify(processed_message)
    return render_template('result.html', prediction=classification, message=user_message)

if __name__ == '__main__':
    app.run(debug=True)
