import re
import string

from flask import request
from bs4 import BeautifulSoup
from flask_limiter.util import get_remote_address


def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()


def stemming(content, port_stem, stop_words, char_to_replace=None):
    content = strip_html(content)  # eliminarea tag-urilor html din text
    content = re.sub('-', '_', content)  # înlocuirea caracterului
    content = re.sub(r'http\S+', '', content)  # eliminarea adreselor web
    content = re.sub(r'[^\w\s]', '', content)  # eliminarea caracterelor speciale, "_" nu este inclus
    content = content.lower()  # transformarea tuturor cuvintelor în litere mici
    content = content.split()  # împărțirea textului, folosind ca delimitator spațiul între 2 cuvinte și trecerea acestora într-o listă
    content = [port_stem.stem(word) for word in content if not word in stop_words and len(word) > 1]  # aplicarea metodei stemming (trecerea cuvântului în forma de bază)
    content = ' '.join(content)  # transformarea listei înapoi într-un șir de caractere
    if char_to_replace is not None:
        content = content.translate(str.maketrans(char_to_replace))

    return content


def check_url(link):  # django url validation regex
    # am extras functia din django, pentru a nu-l mai importa
    regex = re.compile(
        r'^(?:http|ftp)s?://'  # http:// sau https:// sau ftp
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domeniu
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...sau un ip
        r'(?::\d+)?'  # port optional
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)

    return re.match(regex, link) is not None


def isfloat(value):
    if value == "NaN":  # in python "NaN" este considerat float
        return False
    try:
        float(value)
        return True
    except ValueError:
        return False


def get_rate_limit():  # transmitem limitele impuse api-ului, difera daca se ofera un key
    if request.remote_addr == '127.0.0.1':  # nu limitam daca sunt requesturi locale
        return None
    return "50 per hour"
