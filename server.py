import spacy
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
from customFocalLoss import focal_loss_fixed, focal_loss
app = Flask("server")


app = Flask("server")
CORS(app)
# When loading the model
custom_objects = {
    "Custom>focal_loss_fixed": focal_loss_fixed,
    "Custom>focal_loss": focal_loss,
    "Custom>focal_loss_fn": focal_loss(gamma=2.0, alpha=0.25)
}

w2v_modelNN = tf.keras.models.load_model('w2v_model.keras', custom_objects=custom_objects)

w2v_model = Word2Vec.load("word2vec.model")
# Load the IDF dictionary
with open('idf_dict.pkl', 'rb') as f:
    idf_dict = pickle.load(f)



# Load spaCy with minimal components, and enable multi-core processing
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser","entity_linker"])

# Custom preprocessing function that works on a list of texts
def text_preprocess(texts, batch_size=1000, n_process=4):
    cleaned = []

    for doc in tqdm(nlp.pipe(texts, batch_size=batch_size, n_process=n_process)):
        tokens = [
            token.lemma_.lower() for token in doc
            if token.lemma_ != "-PRON-"
            and token.lemma_.lower() not in SPACY_STOPWORDS
            and not token.is_space
            and not token.is_punct
        ]
        cleaned.append(" ".join(tokens))
    
    return cleaned

def predict(text, k=150):
    # Ensure input is a single string
    if not isinstance(text, str):
        raise ValueError("Input should be a single string of text.")
    
    # Preprocess using your batch-friendly function
    cleaned_text = text_preprocess([text])[0]  # returns list; take first element
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
        input_vector = np.zeros((1, k))  # fallback if no known tokens

    # Predict
    probability = w2v_modelNN.predict(input_vector)[0][0]
    print(f"[DEBUG] Probability: {probability:.4f}")
    
    predicted_class = (probability > 0.5).astype(int)  # Threshold can be tuned
    return "spam" if predicted_class == 1 else "ham"

@app.route('/')
def home():
    return "🔥 Hello, Flask is alive! 🔥"

@app.route('/predict', methods=['POST'])
def predictResponse():
    data = request.get_json()
    text = data.get("text", "")
    prediction = predict(text)
    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(debug=True,port=5002)
