import re
import contractions
import pickle
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.models import load_model
from flask import Flask, request, jsonify
from flask_cors import CORS
from gensim.models import Word2Vec
import numpy as np
import spacy
from tqdm import tqdm
import pandas as pd
from spacy.lang.en.stop_words import STOP_WORDS as SPACY_STOPWORDS
import tensorflow as tf
import sys # Import sys for exiting on critical error

# Import custom loss function (assumes customFocalLoss.py is in the root)
from customFocalLoss import focal_loss_fixed, focal_loss

# --- GLOBAL APP SETUP ---
# Renamed from "server" to standard __name__ for Gunicorn compatibility
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


# --- ARTIFACTS LOADING (LOADED ONCE ON STARTUP) ---
try:
    print("Loading ML Models and Artifacts...")
    
    # 1. Define custom objects for Keras model loading
    custom_objects = {
        "Custom>focal_loss_fixed": focal_loss_fixed,
        "Custom>focal_loss": focal_loss,
        "Custom>focal_loss_fn": focal_loss(gamma=2.0, alpha=0.25)
    }

    # 2. Load Keras Model
    w2v_modelNN = tf.keras.models.load_model('w2v_model.keras', custom_objects=custom_objects)

    # 3. Load Word2Vec Model
    w2v_model = Word2Vec.load("word2vec.model")
    
    # 4. Load the IDF dictionary
    with open('idf_dict.pkl', 'rb') as f:
        idf_dict = pickle.load(f)

    # 5. Load spaCy Model (assumes model is downloaded via packages.txt)
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser","entity_linker"])
    
    print("All models loaded successfully.")

except Exception as e:
    # Exit if model loading fails, ensuring the service does not start in a broken state
    print(f"FATAL ERROR: Failed to load models or files. Check paths and dependencies: {e}")
    sys.exit(1)


# --- PREPROCESSING FUNCTION ---
def text_preprocess(texts, batch_size=1, n_process=1):
    cleaned = []
    
    # Use nlp.pipe for efficient processing
    for doc in nlp.pipe(texts, batch_size=batch_size, n_process=n_process):
        tokens = [
            token.lemma_.lower() for token in doc
            if token.lemma_ != "-PRON-"
            and token.lemma_.lower() not in SPACY_STOPWORDS
            and not token.is_space
            and not token.is_punct
        ]
        cleaned.append(" ".join(tokens))
    
    return cleaned

# --- PREDICTION FUNCTION ---
def predict(text, k=150):
    if not isinstance(text, str):
        raise ValueError("Input should be a single string of text.")
    
    # Preprocess text
    cleaned_text = text_preprocess([text])[0]
    tokens = cleaned_text.split()

    # Compute TF-IDF weighted average Word2Vec vector
    vecs = []
    for word in tokens:
        if word in w2v_model.wv:
            weight = idf_dict.get(word, 1.0)
            vecs.append(weight * w2v_model.wv[word])
    
    if vecs:
        input_vector = np.mean(vecs, axis=0).reshape(1, -1)
    else:
        # Fallback: Zero vector if no known tokens
        input_vector = np.zeros((1, k))

    # Predict probability
    probability = w2v_modelNN.predict(input_vector)[0][0]
    # print(f"[DEBUG] Probability: {probability:.4f}") # Console logging is fine in production
    
    # Threshold for classification
    predicted_class = (probability > 0.5).astype(int)
    return "spam" if predicted_class == 1 else "ham"


# --- FLASK ROUTES ---

@app.route('/', methods=['GET'])
def home():
    """Simple health check endpoint."""
    return jsonify({
        "status": "OK",
        "message": "Spam Filtering API is running, ready for POST requests to /predict."
    }), 200

@app.route('/predict', methods=['POST'])
def predictResponse():
    """Prediction endpoint for classifying text."""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "Missing 'text' field in JSON payload"}), 400
            
        text = data.get("text")
        prediction = predict(text)
        
        return jsonify({"prediction": prediction}), 200

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"error": "An internal server error occurred."}), 500

# NOTE: The if __name__ == "__main__": block is removed. 
# Gunicorn handles starting the server.
