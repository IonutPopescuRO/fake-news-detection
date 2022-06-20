from bs4 import BeautifulSoup
import requests
import pandas as pd

headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}

news = []

for page in range(1, 410 + 1):
    html_page = requests.get("https://www.digi24.ro/stiri/actualitate/social?p=" + str(page), headers=headers)
    soup = BeautifulSoup(html_page.text, "html.parser")

    for article in soup.findAll('article'):
        url = article.find('a')
        url = "https://www.digi24.ro" + url.get('href')

        html_article = requests.get(url, headers=headers)
        soup_article = BeautifulSoup(html_article.text, "html.parser")

        get_article = soup_article.find('article')
        title = get_article.find('h1').text.strip()

        content = get_article.find('div', {'class': 'entry data-app-meta data-app-meta-article'}).get_text().strip()
        content_split = content.split("Editor : ")

        content = content_split[0].strip()
        news.append({'title': title, 'text': content, 'label': 1})

df = pd.DataFrame(news, columns=["title", "text", "label"])
df.to_csv('../datasets/ro/2/digi24.csv', encoding='utf-8', index=False)
