# Script for creating pandas.DataFrame pickles out of the article content and metadata from the competition's train.zip training data
# Expects that train.zip and train.json are within the same directory as this script
# To create all dataframes, run `python create_dataframes.py`

import os
import re
from zipfile import ZipFile 
import argparse

import pandas as pd
import numpy as np
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
        contents = get_related_articles_content(zip_file, row.related_articles)
        metadata_article_df.loc[index, 'article_content'] = ' '.join(contents)

    return metadata_article_df


def get_related_articles_content(zip_file, related_articles):
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


def create_fold(metadata_df, train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15):
    """Add a column fold to metadata df"""
    n_rows = metadata_df.shape[0]
    np.random.seed(123)
    indices = np.random.permutation(n_rows)
    train_indices, dev_indices = np.ceil(train_ratio * n_rows).astype(int), np.ceil((1 - test_ratio) * n_rows).astype(
        int)
    training_idx, dev_idx, test_idx = indices[:train_indices], indices[train_indices:dev_indices], indices[
                                                                                                   dev_indices:n_rows]
    metadata_df["fold"] = "train"
    metadata_df.loc[dev_idx, "fold"] = "development"
    metadata_df.loc[test_idx, "fold"] = "test"

    return metadata_df


if __name__ == '__main__':
    PROJECT_DIR = os.path.dirname(os.path.realpath(__file__))

    DATA_FILENAME = 'train.zip'
    DATA_FILEPATH = os.path.join(PROJECT_DIR, DATA_FILENAME)

    METADATA_FILENAME = 'train.json'
    METADATA_FILEPATH = os.path.join(PROJECT_DIR, 'train.json')

    parser = argparse.ArgumentParser()
    parser.add_argument('--create-articles', action='store_true')
    parser.set_defaults(create_all=False)
    args = parser.parse_args()
    create_articles = args.create_articles

    zip_file = ZipFile(DATA_FILEPATH, 'r')

    if create_articles:
        article_df = create_articles_df(zip_file)
        path_to_save = os.path.join(PROJECT_DIR, 'articles_dataframe.pkl')
        article_df.reset_index().to_pickle(path_to_save)

    metadata_df = create_metadata_df(METADATA_FILEPATH)
    path_to_save = os.path.join(PROJECT_DIR, 'metadata_dataframe.pkl')
    metadata_df = create_fold(metadata_df.reset_index())
    metadata_df.to_pickle(path_to_save)

    metadata_articles_df = create_metadata_articles_df(zip_file, metadata_df)
    path_to_save = os.path.join(PROJECT_DIR, 'metadata_articles_dataframe.pkl')
    metadata_articles_df = create_fold(metadata_articles_df.reset_index())
    metadata_articles_df.to_pickle(path_to_save)

    zip_file.close()
    print('Success, Dataframes created!')
