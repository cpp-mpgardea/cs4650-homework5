import pickle
import numpy as np
import pandas as pd
from newsapi.newsapi_client import NewsApiClient
import en_core_web_lg
from collections import Counter
import string
nlp_eng = en_core_web_lg.load()
newsapi = NewsApiClient (api_key='7fcbd57deb4946a39be107ba64aca14a')

pos_tag = ['NOUN', 'VERB', 'PROPN']

def get_keywords_eng(content):
  doc = nlp_eng(content)
  result = []
  for token in doc:
    if (token.text in nlp_eng.Defaults.stop_words or token.text in string.punctuation):
      continue
    if (token.pos_ in pos_tag):
      result.append(token.text)
  return result

articles = newsapi.get_everything(q='coronavirus', language='en', from_param='2021-02-24', to='2021-03-23', sort_by='relevancy', page=1)

i = 2
while i <= 5:
  article = newsapi.get_everything(q='coronavirus', language='en', from_param='2021-02-24', to='2021-03-23', sort_by='relevancy', page=i)
  articles['articles'].extend(article['articles'])
  i+=1

filename = 'articlesCOVID.pckl'
pickle.dump(articles, open(filename, 'wb'))
filename = 'articlesCOVID.pckl'
loaded_model = pickle.load(open(filename, 'rb'))
filepath = 'articlesCOVID.pckl'
pickle.dump(loaded_model, open(filepath, 'wb'))

dados = []

for i, article in enumerate(articles['articles']):
    title = article['title']
    description = article['description']
    date = article['publishedAt']
    content = article['content']
    dados.append({'title':title, 'date':date, 'desc':description, 'content':content})
df = pd.DataFrame(dados)
df = df.dropna()
df.head()

results = []

for content in df.content.values:
    results.append([('#' + x[0]) for x in Counter(get_keywords_eng(content)).most_common(5)])
df['keywords'] = results

filename = 'articlesCOVID.pckl'
pickle.dump(df, open(filename, 'wb'))
filename = 'articlesCOVID.pckl'
loaded_model = pickle.load(open(filename, 'rb'))
filepath = 'articlesCOVID.pckl'
pickle.dump(loaded_model, open(filepath, 'wb'))

print(df)