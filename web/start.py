import pickle
from langdetect import detect

import flask
from flask import jsonify, render_template
from flask_limiter import Limiter
from nltk import PorterStemmer
from nltk.stem.snowball import RomanianStemmer
from nltk.corpus import stopwords

from functions import *

from newspaper import Article

app = flask.Flask(__name__)
app.config.from_object("config.Config")

limiter = Limiter(app, key_func=get_remote_address)

models_list = [
    'logistic_regression',
    'decision_tree_classification',
    'gradient_boosting_classifier',
    'random_forest_classifier'
]

ro_model = []
en_model = []

for i in range(len(models_list)):
    ro_model.append(pickle.load(open('../trained_models/ro/'+models_list[i]+'.pkl', 'rb')))
    en_model.append(pickle.load(open('../trained_models/en/'+models_list[i]+'.pkl', 'rb')))

vectorizer = pickle.load(open('../trained_models/en/vectorizer.pkl', 'rb'))
vectorizer_ro = pickle.load(open('../trained_models/ro/vectorizer.pkl', 'rb'))

char_to_replace = {
    'ă': 'a',
    'â': 'a',
    'î': 'i',
    'ș': 's',
    'ț': 't',

    'ş': 's',
    'ţ': 't'
}

port_stem = PorterStemmer()
port_stem_ro = RomanianStemmer()

punctuation = string.punctuation.replace("_", "")

stop_words = stopwords.words('english')
stop_words.extend(punctuation)

stop_words_ro = stopwords.words('romanian')
stop_words_ro.extend(punctuation)


@app.route('/', methods=['GET'])
def home():
    status = {"name": "Fake News Detection", "version": "v1.0", "status": "online"}  # demo status
    return render_template('index.html', status=str(status))


@app.route('/api/', methods=['POST'])
@limiter.limit(get_rate_limit)
def api():
    result = {}

    lang = "en"
    text = request.form['text']
    param_type = request.form['type']

    if param_type == "url":
        article = Article(text)
        article.download()
        article.parse()
        text = article.title + " " + article.text

        if article.meta_lang == 'ro':
            lang = "ro"
    else:
        if detect(text) == 'ro':
            lang = 'ro'

    if lang == 'ro':
        nlp_text = stemming(text, port_stem_ro, stop_words_ro, char_to_replace)
        vectorized = vectorizer_ro.transform([nlp_text])

        result["lang"] = 'ro'
        for i in range(len(models_list)):
            t = ro_model[i].predict(vectorized)
            result[models_list[i]] = "False" if t[0] == 0 else "True"
    else:
        nlp_text = stemming(text, port_stem, stop_words)
        vectorized = vectorizer.transform([nlp_text])

        result["lang"] = 'en'
        for i in range(len(models_list)):
            t = en_model[i].predict(vectorized)
            result[models_list[i]] = "False" if t[0] == 0 else "True"

    result['nlp_text'] = nlp_text

    return jsonify(result)


@app.errorhandler(429)
def resource_not_found(e):
    return jsonify(error=str(e)), 429


app.run(host='0.0.0.0', port=5000)
