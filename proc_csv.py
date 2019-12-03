import pickle
import pandas as pd
import os
import string
import numpy as np


def get_claim_and_body(str):
    """
    separates the body and the claim in a single entry
    """
    summary = []
    for i, char in enumerate(str):
        if char != '.':
            summary.append(char)

        elif i > 0.2 * len(str):
            break
        elif char == '.':
            break
    summary = convert(summary)
    summary = summary.translate(str.maketrans('', '', string.punctuation))

    body = str[i:len(str)]
    body = body.translate(str.maketrans('', '', string.punctuation))

    return summary, body


def convert(s):
    # initialization of string to ""
    str1 = ""

    # using join function join the list s by
    # separating words by str1
    return (str1.join(s))




df = pd.read_pickle("preprocessed_training_dataframe.pkl")
# we expand this df by one col, and then we fill it with the real summary
print(len(df.index))
col_names = ['label', 'claim', 'body']
my_df = pd.DataFrame(columns=col_names)

for index, row in df.iterrows():
    # modify the original body
    claim, body = get_claim_and_body(row['X'])
    new_df = pd.DataFrame([[row['label'], claim, body]], columns=col_names)
    my_df = my_df.append(new_df, ignore_index=True)
    if index == 100:
        print("here")
        break

print(my_df)
path_to_save = "preprocessed_claim_body.pkl"
my_df.to_pickle(path_to_save)



data = pd.read_pickle(os.path.join("", "preprocessed_claim_body.pkl"))
print(data)

