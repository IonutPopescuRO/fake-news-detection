import time

import requests
from requests.adapters import HTTPAdapter, Retry

import pandas as pd
from bs4 import BeautifulSoup

"""
43 - social
44 - monden
46 - sport
"""

headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}

news = []
offset = 0

s = requests.Session()

retries = Retry(total=20,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504])

s.mount('https://', HTTPAdapter(max_retries=retries))

while True:
    try:
        blog_json = s.get(
            "https://www.timesnewroman.ro/wp-json/wp/v2/posts/?categories=43,44,46,57&per_page=100&offset=" + str(offset),
            timeout=10).json()
    except Exception as e:
        time.sleep(10)
        continue

    print(offset)
    offset += 100
    if len(blog_json) == 0:
        break
    for entry in blog_json:
        # date = entry['date_gmt']
        title = entry['title']['rendered']
        content = entry['content']['rendered']

        text = BeautifulSoup(content, "html.parser").get_text()
        news.append({'title': title, 'text': text, 'label': 0})

df = pd.DataFrame(news, columns=["title", "text", "label"])
df.to_csv('../datasets/ro/2/timesnewroman3.csv', encoding='utf-8', index=False)
