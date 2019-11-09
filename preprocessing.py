import os
import re
import string
import unicodedata

import pandas as pd
from tqdm import tqdm_notebook as tqdm
import contractions
import inflect
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer


def preprocess(data_df):
    """Create a dataframe whose contents are cleaned, stemmed, and lemmatized"""
    data_cp = data_df.copy()
    for i, row in tqdm(data_cp.iterrows(), total=len(data_cp), desc='Preprocessing dataframe contents'):

        article_content = _clean(row.article_content)
        row.article_content = _tokenize_stem_lem_join(article_content)
        row.claimant = _tokenize_stem_lem_join(row.claimant)
        row.claim = _tokenize_stem_lem_join(row.claim)

        data_cp.loc[i] = row

    return data_cp


def _clean(text):
    """Cleans text with an aggregate of cleaning methods"""
    text = _remove_between_square_brackets(text)
    text = _replace_contractions(text)
    
    words = nltk.word_tokenize(text)
    words = _remove_non_ascii(words)
    words = _to_lowercase(words)
    words = _remove_punctuation(words)
    words = _replace_numbers(words)
    # words = remove_stopwords(words)

    return ' '.join(words)


def _remove_between_square_brackets(text):
    """Remove all text between squared brakets"""
    return re.sub('\[[^]]*\]', '', text)


def _replace_contractions(text):
    """Replace contractions in string of text"""
    return contractions.fix(text)


def _remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words


def _to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words


def _remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words


def _replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            try:
                new_word = p.number_to_words(word)
                new_words.append(new_word)
            except:
                pass
        else:
            new_words.append(word)
    return new_words


def _remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words


def _tokenize_stem_lem_join(text):
    """Stem and lemmatize text by tokenizing and joining back together"""
    words = nltk.word_tokenize(text)
    words = _stem_words(words)
    words = _lemmatize_verbs(words)
    return ' '.join(words)
    

def _stem_words(words):
    """Stem words in list of tokenized words"""
    stemmer = LancasterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems


def _lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas


def create_dataframe_for_training(data):
    """Creates a dataframe for training by concatenating claimant, claim and article content and
    copying labels to a new dataframe"""
    feature_column_name = 'X'
    data_cp = data[['label']].copy()
    for i, row in tqdm(data.iterrows(), total=len(data)):
        all_features = f'{row.claimant} {row.claim} {row.article_content}'
        data_cp.loc[i, feature_column_name] = all_features

    return data_cp


if __name__ == '__main__':

    # Required for stemming, lemming, and removing stopwords
    nltk.download('punkt')
    nltk.download('wordnet')

    PROJECT_DIR = os.path.dirname(os.path.realpath(__file__))
    DATA_DIR = os.path.join(PROJECT_DIR, 'data')
    DATA_FILEPATH = os.path.join(DATA_DIR, 'metadata_articles_dataframe.pkl')

    data = pd.read_pickle(DATA_FILEPATH)
    data = data[['claim', 'claimant', 'article_content', 'label']]

    data_clean = preprocess(data)
    X = create_dataframe_for_training(data_clean)
    path_to_save = os.path.join(DATA_DIR, 'metadata_articles_dataframe_preprocessed.pkl')
    X.to_pickle(path_to_save)
    