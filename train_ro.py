import pandas as pd
import re
import pickle
import string
# import nltk

from nltk.corpus import stopwords
from nltk.stem.snowball import RomanianStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from bs4 import BeautifulSoup

classifiers = ["Logistic Regression", "Decision Tree Classification", "Gradient Boosting Classifier",
               "Random Forest Classifier"]

port_stem = RomanianStemmer()

# nltk.download('stopwords')
stop_words = stopwords.words('romanian')
stop_words.extend(string.punctuation.replace("_", ""))


char_to_replace = {
    'ă': 'a',
    'â': 'a',
    'î': 'i',
    'ș': 's',
    'ț': 't',

    'ş': 's',
    'ţ': 't'
}


def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()


def text_processing(content):
    content = strip_html(content)  # eliminarea tag-urilor html din text
    content = re.sub('-', '_', content)  # înlocuirea caracterului
    content = re.sub(r'http\S+', '', content)  # eliminarea adreselor web
    content = re.sub(r'[^\w\s]', '', content)  # eliminarea caracterelor speciale, "_" nu este inclus
    content = content.lower()  # transformarea tuturor cuvintelor în litere mici
    content = content.split()  # împărțirea textului, folosind ca delimitator spațiul între 2 cuvinte și trecerea acestora într-o listă
    content = [port_stem.stem(word) for word in content if not word in stop_words and len(word) > 1]  # aplicarea metodei stemming (trecerea cuvântului în forma de bază)
    content = ' '.join(content)  # transformarea listei înapoi într-un șir de caractere
    content = content.translate(str.maketrans(char_to_replace))

    return content


def get_model(x):
    match x:
        case 0:  # Logistic Regression
            return LogisticRegression()
        case 1:  # Decision Tree Classification
            return DecisionTreeClassifier()
        case 2:  # Gradient Boosting Classifier
            return GradientBoostingClassifier(random_state=0)
        case 3:  # Random Forest Classifier
            return RandomForestClassifier(random_state=0)
        case _:  # Unknown case
            return None


digi24_dataset = pd.read_csv('datasets/ro/2/digi24.csv')
timesnewroman_dataset = pd.read_csv('datasets/ro/2/timesnewroman.csv')

news_dataset = pd.concat([digi24_dataset, timesnewroman_dataset])

news_dataset = news_dataset.fillna('')

news_dataset['content'] = (news_dataset['title'] + " " + news_dataset['text']).apply(text_processing)

X = news_dataset['content']
Y = news_dataset['label']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)

vectorized = TfidfVectorizer(use_idf=True)
xv_train = vectorized.fit_transform(X_train)
xv_test = vectorized.transform(X_test)

pickle.dump(vectorized, open('trained_models/ro/vectorizer.pkl', 'wb'))
print("Saved vectorizer.\n")

for i in range(len(classifiers)):
    model = get_model(i)
    model.fit(xv_train, Y_train)

    pickle.dump(model, open('trained_models/ro/' + classifiers[i].lower().replace(' ', '_') + '.pkl', 'wb'))

    train_prediction = model.predict(xv_train)
    test_prediction = model.predict(xv_test)

    print(classifiers[i])
    print("Test data accuracy: ",
          model.score(xv_test, Y_test))
    print("Training data accuracy: ", accuracy_score(train_prediction, Y_train))
    print()
    print(classification_report(Y_test, test_prediction))

print("Done.")
