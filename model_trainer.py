"""
News Article Categorizer - Model Trainer
This script trains a Naive Bayes classifier on the BBC news dataset
and saves the model and vectorizers for later use.
"""

import pandas as pd
import numpy as np
import nltk
import pickle
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# Download required NLTK resources
print("Downloading NLTK resources...")
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

def preprocess_text(text):
    """Preprocess text by tokenizing and removing stopwords"""
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if word not in stop_words]
    return tokens

def train_model():
    """Train the Naive Bayes model on BBC dataset"""
    
    # Load dataset from BBC News Train.csv
    csv_path = 'BBC News Train.csv'
    
    if not os.path.exists(csv_path):
        print(f"Error: Dataset not found at {csv_path}")
        print("Please make sure 'BBC News Train.csv' is in the project folder.")
        return False
    
    print("Loading dataset...")
    df = pd.read_csv(csv_path)
    
    # Display unique categories
    print(f"Categories in dataset: {df['Category'].unique()}")
    print(f"Dataset shape: {df.shape}")
    print(f"Sample data:\n{df.head()}")
    
    # Preprocess the text data
    print("\nPreprocessing text...")
    df['processed_content'] = df['Text'].apply(preprocess_text)
    
    # Join tokens back to string for vectorization
    df['processed_text'] = df['processed_content'].apply(' '.join)
    
    # Create BoW and TF-IDF vectorizers
    print("Creating BoW vectorizer...")
    bow_vectorizer = CountVectorizer()
    X_bow = bow_vectorizer.fit_transform(df['processed_text'])
    
    print("Creating TF-IDF vectorizer...")
    tfidf_vectorizer = TfidfVectorizer()
    X_tfidf = tfidf_vectorizer.fit_transform(df['processed_text'])
    
    y = df['Category']
    
    # Split data
    X_train_bow, X_test_bow, y_train, y_test = train_test_split(
        X_bow, y, test_size=0.2, random_state=42
    )
    X_train_tfidf, X_test_tfidf, _, _ = train_test_split(
        X_tfidf, y, test_size=0.2, random_state=42
    )
    
    # Train BoW model
    print("Training BoW model...")
    nb_model_bow = MultinomialNB()
    nb_model_bow.fit(X_train_bow, y_train)
    
    # Evaluate BoW model
    y_pred_bow = nb_model_bow.predict(X_test_bow)
    print(f"\nBoW Model Accuracy: {accuracy_score(y_test, y_pred_bow):.4f}")
    print("BoW Classification Report:")
    print(classification_report(y_test, y_pred_bow))
    
    # Train TF-IDF model
    print("Training TF-IDF model...")
    nb_model_tfidf = MultinomialNB()
    nb_model_tfidf.fit(X_train_tfidf, y_train)
    
    # Evaluate TF-IDF model
    y_pred_tfidf = nb_model_tfidf.predict(X_test_tfidf)
    print(f"\nTF-IDF Model Accuracy: {accuracy_score(y_test, y_pred_tfidf):.4f}")
    print("TF-IDF Classification Report:")
    print(classification_report(y_test, y_pred_tfidf))
    
    # Save models and vectorizers
    print("\nSaving models...")
    
    models = {
        'bow_vectorizer': bow_vectorizer,
        'tfidf_vectorizer': tfidf_vectorizer,
        'nb_model_bow': nb_model_bow,
        'nb_model_tfidf': nb_model_tfidf,
        'categories': list(df['Category'].unique())
    }
    
    with open('news_model.pkl', 'wb') as f:
        pickle.dump(models, f)
    
    print("Model saved successfully as 'news_model.pkl'")
    return True

def predict_category(text, model_type='bow'):
    """Make prediction on new text"""
    
    # Load saved model
    if not os.path.exists('news_model.pkl'):
        print("Error: Model file not found. Please run train_model() first.")
        return None
    
    with open('news_model.pkl', 'rb') as f:
        models = pickle.load(f)
    
    # Preprocess input text
    processed_text = ' '.join(preprocess_text(text))
    
    # Select model and vectorizer
    if model_type == 'bow':
        vectorizer = models['bow_vectorizer']
        model = models['nb_model_bow']
    else:
        vectorizer = models['tfidf_vectorizer']
        model = models['nb_model_tfidf']
    
    # Transform and predict
    text_vector = vectorizer.transform([processed_text])
    prediction = model.predict(text_vector)[0]
    
    return prediction

if __name__ == "__main__":
    success = train_model()
    if success:
        # Test prediction
        test_text = "Artificial intelligence is revolutionizing the tech industry"
        result = predict_category(test_text, 'bow')
        print(f"\nTest prediction for: '{test_text}'")
        print(f"Predicted category: {result}")

