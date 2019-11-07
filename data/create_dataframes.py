# Script for creating pandas.DataFrame pickles out of the article content and metadata from the competition's train.zip training data
# Expects that train.zip and train.json are within the same directory as this script
# To create all dataframes, run `python create_dataframes.py`

import os
import re
from zipfile import ZipFile 
import argparse

import pandas as pd
from tqdm import tqdm


def create_articles_df(zip_file):
    """Creates a pandas.DataFrame from articles within the competition's train.zip file"""
    articles_df = pd.DataFrame(columns=['text'])
    article_relative_filepaths = [fp for fp in zip_file.namelist() if '.txt' in fp]

    for filepath in tqdm(article_relative_filepaths, desc='Creating articles df'):
        article_id = re.findall(r'\d+', filepath)[0]
        content = read_article_content(zip_file, filepath)
    
        articles_df.loc[article_id, 'text'] = content

    return articles_df


def read_article_content(zip_file, article_filepath, encoding='utf-8'):
    """Reads article content by opening and decoding it"""
    with zip_file.open(article_filepath) as f:
        content = f.read()
        decoded_content = content.decode(encoding)
        return decoded_content


def create_metadata_articles_df(zip_file, metadata_df):
    """Creates pandas.DataFrame by deep copying metadata dataframe and concatenating claims' related article content into a new column."""
    metadata_article_df = metadata_df.copy()
    metadata_article_df.article_content = ''

    for index, row in tqdm(metadata_article_df.iterrows(), desc='Creating metadata article df', total=len(metadata_df)):
        contents = get_related_articles_content(row.related_articles)
        metadata_article_df.loc[index, 'article_content'] = ' '.join(contents)

    return metadata_article_df


def get_related_articles_content(related_articles):
    """Gets all related articles content by reading each related article"""
    contents = []
    for article_id in related_articles:
        filepath = f'train_articles/{article_id}.txt'
        content = read_article_content(zip_file, filepath)
        contents.append(content)
    return contents
    

def create_metadata_df(metadata_filepath):
    """Creates a pandas.DataFrame out of the metadata file"""
    metadata_df = pd.read_json(metadata_filepath)
    metadata_df = metadata_df.set_index('id', drop=True)
    return metadata_df


if __name__ == '__main__':
    PROJECT_DIR = os.path.dirname(os.path.realpath(__file__))

    DATA_FILENAME = 'train.zip'
    DATA_FILEPATH = os.path.join(PROJECT_DIR, DATA_FILENAME)

    METADATA_FILENAME = 'train.json'
    METADATA_FILEPATH = os.path.join(PROJECT_DIR, 'train.json')

    zip_file = ZipFile(DATA_FILEPATH, 'r')

    article_df = create_articles_df(zip_file)
    path_to_save = os.path.join(PROJECT_DIR, 'articles_dataframe.pkl')
    article_df.to_pickle(path_to_save)

    metadata_df = create_metadata_df(METADATA_FILEPATH)
    path_to_save = os.path.join(PROJECT_DIR, 'metadata_dataframe.pkl')
    metadata_df.to_pickle(path_to_save)

    metadata_articles_df = create_metadata_articles_df(zip_file, metadata_df)
    path_to_save = os.path.join(PROJECT_DIR, 'metadata_article_dataframe.pkl')
    metadata_articles_df.to_pickle(path_to_save)

    zip_file.close()
