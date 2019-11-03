"""This script is a AdaBoost classifier submission for the Leaders Prize competition.
It reads in a dataset and creates a predictions file.
"""
import json
import os
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd
import pickle
import os
import numpy as np

model = pickle.load(open("/usr/src/naive_bayes_w_articles.sav", 'rb'))
vectorizer = pickle.load(open('usr/src/vectorizer.pk', 'rb'))

# These are the file paths where the validation/test set will be mounted (read only)
# into your Docker container.
METADATA_FILEPATH = '/usr/local/dataset/metadata.json'
ARTICLES_FILEPATH = '/usr/local/dataset/articles/'

# This is the filepath where the predictions should be written to.
PREDICTIONS_FILEPATH = '/usr/local/predictions.txt'

# Create X data for texting model
X_test = pd.read_json(METADATA_FILEPATH)
X_test_related_articles = pd.DataFrame()

# Add article content to this dataset
for idx, row in (X_test.iterrows()):
    X_test_related_articles.at[idx, "content_article"] = ""
    for a in row.related_articles:
        X_test_related_articles.at[idx, "content_article"] +=  " " + open(ARTICLES_FILEPATH + str(a) + ".txt", "r").read()  
        
X_test_related_articles = pd.concat([X_test, X_test_related_articles], axis=1)


def find_similar(tfidf_matrix, index, top_n = 5):
    cosine_similarities = linear_kernel(tfidf_matrix[index:index+1], tfidf_matrix).flatten()
    related_docs_indices = [i for i in cosine_similarities.argsort()[::-1] if i != index]
    return [(index, cosine_similarities[index]) for index in related_docs_indices][0:top_n]


def get_top_5_similarity(df, vect):

    top_5_list = []
    
    for idx, row in (df.iterrows()):
        content = row.content_article
        claim = row.claim
        claimant = row.claimant
        corpus = np.append([claim], content.replace("\n", ".").split("."))
        tfidf_trans = vect.transform(corpus)
        top_5 = find_similar(tfidf_trans, 0, top_n=10)
        
        top_5_sentences = claimant + " " + claim
        for a, (i,score) in enumerate(top_5):
            top_5_sentences += (corpus[i]) + " "
            
        out = vect.transform([top_5_sentences]).toarray()[0]
        top_5_list.append(out)
        
    return df, np.array(top_5_list)

df = pd.DataFrame()
df, top_5_list = get_top_5_similarity(X_test_related_articles, vectorizer)

# Create a predictions file.
print('\nWriting predictions to:', PREDICTIONS_FILEPATH)
with open(PREDICTIONS_FILEPATH, 'w') as f:
    for x, top_5_vec in zip(df.iterrows(),top_5_list) :
        i = x[1]['id']
        val = (model.predict(top_5_vec.reshape(1, -1)))[0]
        f.write('%d,%d\n' % (i,val))

print('Finished writing predictions.')
