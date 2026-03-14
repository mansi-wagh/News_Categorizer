# News Article Categorizer

A beautiful web application that automatically classifies news articles into categories using Machine Learning (Naive Bayes Classifier).

## Categories
- Entertainment
- Business
- Sport
- Politics
- Tech

## Project Structure
```
news-categorizer/
├── app.py              # Flask backend server
├── model_trainer.py    # Script to train and save the model
├── requirements.txt    # Python dependencies
├── README.md           # This file
├── static/
│   └── style.css       # Beautiful CSS styling
└── templates/
    └── index.html      # Frontend user interface
```

## How to Train the Model

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Make sure BBC News dataset is in the project folder
The dataset file should be named: `BBC News Train.csv`

### Step 3: Run the training script
```bash
python model_trainer.py
```

This will:
- Load the BBC News dataset
- Preprocess the text (tokenization, remove stopwords)
- Train both BoW and TF-IDF models using Naive Bayes
- Save the trained model as `news_model.pkl`

### Step 4: Run the Flask server
```bash
python app.py
```

### Step 5: Open in Browser
Go to: http://127.0.0.1:5000

## Features
- ✨ Beautiful, modern UI with gradient design
- 📝 Text input for news articles/headlines
- 🎯 Real-time category prediction
- 📊 Probability bars showing confidence for each category
- 🔄 Choice between BoW and TF-IDF models
- 📱 Responsive design for mobile devices

## Usage
1. Enter any news text in the text box
2. Select model type (BoW or TF-IDF)
3. Click "Categorize News"
4. View the predicted category and probability distribution

