import time

# pip install googletrans==4.0.0-rc1
from googletrans import Translator
import pandas as pd

translator = Translator()


def translate(content):
    content = str(content)
    if len(content) > 5000:
        return None

    time.sleep(1)
    try:
        translated = translator.translate(content, src='en', dest='ro')
        return translated.text
    except:
        return ""


news_dataset = pd.read_csv('../datasets/en/1/train.csv')
# print(test_results.head())

news_dataset['content'] = (news_dataset['title'] + ". " + news_dataset['text']).apply(translate)

vector_not_null = news_dataset['content'].notnull()
df_not_null = news_dataset[vector_not_null]
df_not_null = df_not_null.drop(['id', 'title', 'author', 'text'], axis=1)

df_not_null.to_csv('../datasets/ro/1/train.csv', encoding='utf-8', index=False)
