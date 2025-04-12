import spacy
import re
import contractions
import pickle
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.models import load_model
from flask import Flask, request, jsonify
from flask_cors import CORS
app = Flask("server")


app = Flask("server")
CORS(app)
# Load model
model = load_model('my_model.keras')

# Load fitted TF-IDF vectorizer (must be saved during training)
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Load Spacy model once
nlp = spacy.load("en_core_web_sm")

STOPWORDS = set(stopwords.words("english"))

def text_preprocess(text):
    text = text.lower()
    text = contractions.fix(text)
    text = re.sub(r'[^a-zA-Z\s!$]', " ", text)
    text = re.sub(r'\s+', " ", text)
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if token.lemma_.lower() not in STOPWORDS and not token.is_space and not token.is_punct]
    return " ".join(tokens)

def predict(text):
    clean_text = text_preprocess(text)
    input_vector = vectorizer.transform([clean_text]).toarray()
    prediction = model.predict(input_vector)
    predicted_class = (prediction > 0.5).astype(int)
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
