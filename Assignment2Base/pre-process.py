import re
import numpy as np
import pandas as pd

import os
import requests
import zipfile


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def download_data(data_path):
    toy_data_path = os.path.join(data_path, 'fever_data.zip')
    toy_data_url_id = "1wArZhF9_SHW17WKNGeLmX-QTYw9Zscl1"
    toy_url = "https://docs.google.com/uc?export=download"

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    if not os.path.exists(toy_data_path):
        print("Downloading FEVER data splits...")
        with requests.Session() as current_session:
            response = current_session.get(toy_url,
                                           params={'id': toy_data_url_id},
                                           stream=True)
        save_response_content(response, toy_data_path)
        print("Download completed!")

        print("Extracting dataset...")
        with zipfile.ZipFile(toy_data_path) as loaded_zip:
            loaded_zip.extractall(data_path)
        print("Extraction completed!")


def pre_process(dataset, filename):  # clean the dataset
    dataset.drop(dataset.columns[0], axis=1, inplace=True)  # remove first column of dataframe containing numbers
    dataset.drop(['ID'], axis=1, inplace=True)
    # remove numbers before each evidence
    dataset['Evidence'] = dataset['Evidence'].apply(lambda x: re.sub(r'^\d+\t', '', x))
    # remove everything after the period
    dataset['Evidence'] = dataset['Evidence'].apply(lambda x: re.sub(r' \..*', ' .', x))
    # remove round brackets and what they contain
    dataset['Evidence'] = dataset['Evidence'].apply(lambda x: re.sub(r'-LRB-.*-RRB-', '', x))
    # remove square brackets and what they contain
    dataset['Evidence'] = dataset['Evidence'].apply(lambda x: re.sub(r'-LSB-.*-RSB-', '', x))

    n_before = dataset.shape[0]
    # removes instances longer than a threshold on evidence
    dataset = dataset[dataset['Evidence'].str.split().str.len() <= 100]
    # remove all rows where there are single brackets in the evidence
    dataset = dataset[~dataset['Evidence'].str.contains('|'.join(['-LRB-', '-LSB-', '-RRB-', '-RSB-']))]
    n_after = dataset.shape[0]

    # removes punctuation and excessive spaces
    dataset = dataset.applymap(lambda x: re.sub(r'[^\w\s]', '', x))
    dataset = dataset.applymap(lambda x: re.sub(r' +', ' ', x))
    dataset = dataset.applymap(lambda x: re.sub(r'^ +', '', x))
    dataset = dataset.applymap(lambda x: x.lower())

    labels = {'supports': 1, 'refutes': 0}
    dataset = dataset.replace({'Label': labels})
    # removes rows with empty elements
    dataset = dataset[dataset['Evidence'] != '']
    dataset = dataset[dataset['Claim'] != '']
    dataset = dataset[dataset['Label'] != '']



    rem_elements = n_before - n_after
    print(f"Removed {rem_elements}\t ({100 * rem_elements / n_before:.2F}%)"
          f" elements because of inconsistency on {filename}")
    return dataset


# download_data('dataset')

for file in os.listdir("dataset"):
    dataset_cleaned = pre_process(pd.read_csv("dataset/" + file, sep=','), file)
    dataset_cleaned.to_csv(os.path.join("dataset_cleaned", file))
