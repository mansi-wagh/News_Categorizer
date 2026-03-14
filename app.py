"""
News Article Categorizer - Flask Backend
This is the Flask application that serves the frontend and provides
an API endpoint for news categorization predictions.
"""

from flask import Flask, render_template, request, jsonify
import pickle
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

app = Flask(__name__)

# Download required NLTK resources
print("Downloading NLTK resources...")
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

# Global variables to store model and vectorizers
models = None

def load_model():
    """Load the trained model from pickle file"""
    global models
    
    if not os.path.exists('news_model.pkl'):
        print("Error: Model file 'news_model.pkl' not found!")
        print("Please run 'python model_trainer.py' first to train and save your model.")
        return False
    
    print("Loading trained model...")
    with open('news_model.pkl', 'rb') as f:
        models = pickle.load(f)
    
    print(f"Model loaded successfully!")
    print(f"Categories: {models['categories']}")
    return True

def preprocess_text(text):
    """Preprocess text by tokenizing and removing stopwords"""
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def predict_category(text, model_type='bow'):
    """Make prediction on input text"""
    
    if models is None:
        return None, "Model not loaded"
    
    # Preprocess input text
    processed_text = preprocess_text(text)
    
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
    
    # Get prediction probabilities
    probabilities = model.predict_proba(text_vector)[0]
    prob_dict = {cat: float(prob) for cat, prob in zip(model.classes_, probabilities)}
    
    return prediction, prob_dict

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html', categories=models['categories'])

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for making predictions"""
    data = request.get_json()
    
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    text = data['text']
    model_type = data.get('model_type', 'bow')
    
    if not text.strip():
        return jsonify({'error': 'Empty text provided'}), 400
    
    try:
        prediction, probabilities = predict_category(text, model_type)
        
        if prediction is None:
            return jsonify({'error': 'Model not loaded properly'}), 500
        
        return jsonify({
            'success': True,
            'prediction': prediction,
            'probabilities': probabilities,
            'text': text
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/categories', methods=['GET'])
def get_categories():
    """API endpoint to get available categories"""
    if models is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'categories': models['categories']
    })

if __name__ == '__main__':
    # Load model on startup
    if load_model():
        print("\n" + "="*50)
        print("Server running at: http://127.0.0.1:5000")
        print("="*50 + "\n")
        app.run(debug=True)
    else:
        print("\nFailed to load model. Please train the model first.")
        print("Run: python model_trainer.py")

