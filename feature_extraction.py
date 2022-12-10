import pyedflib
import numpy as np
from tqdm.notebook import tqdm
import pandas as pd
import pyeeg
import warnings
import multiprocessing
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


# source: https://www.hindawi.com/journals/cin/2011/406391/
# Вычисление основных признаков, функция взята из статьи
def features(mat):
    Kmax = 5
    Tau = 4
    DE = 10
    M = 10
    R = 0.3
    Band = np.arange(1, 86)
    Fs = 173
    DFA = pyeeg.dfa(mat)
    HFD = pyeeg.hfd(mat, Kmax)
    SVD_Entropy = pyeeg.svd_entropy(mat, Tau, DE)
    Fisher_Information = pyeeg.fisher_info(mat, Tau, DE)
    PFD = pyeeg.pfd(mat)

    return (DFA, HFD, SVD_Entropy, Fisher_Information, PFD)


def end_extraction(response):
    for el in response:
        data.loc[len(data.index)] = el
    data.to_csv('dataset.csv')


def features_extraction(sample):
    sample_features = {'name': sample}
    data_sample = pyedflib.EdfReader(f"{dataset_path}/{sample}")
    labels = data_sample.getSignalLabels()
    for i, label in enumerate(labels):
        if label in chanels:
            f_values = features(data_sample.readSignal(i))
            for j in range(1, 6):
                sample_features[f'{label}_f{j}'] = f_values[j - 1]

    sample_features['target'] = 1 if sample in seizures_targets else 0
    data_sample.close()
    print(f"Sample {sample} is completed")
    return sample_features


if __name__ == '__main__':
    dataset_path = "../EEG/physionet.org/files/chbmit/1.0.0/"

    # Получение таргетов/home/didges/ds_learning/seizures/dataset_window.csv
    all_targets = open(f"{dataset_path}/RECORDS", "r").read().split("\n")[:-1]
    seizures_targets = open(f"{dataset_path}/RECORDS-WITH-SEIZURES", "r").read().split("\n")[:-1]

    # Определение уникальных каналов
    chanels = set()
    for sample in tqdm(all_targets):
        data_sample = pyedflib.EdfReader(f"{dataset_path}/{sample}")
        labels = data_sample.getSignalLabels()
        for ch in labels:
            chanels.add(ch)
        data_sample.close()

    drop = ['-', '.', 'CP2-Ref', 'FC1-Ref', 'CP6-Ref', 'FC6-Ref',
            'CP1-Ref', 'FC2-Ref', 'FC5-Ref', 'CP5-Ref', 'VNS', 'ECG']
    for i in drop:
        chanels.remove(i)

    # Создание датафрейма
    data_features = np.array([[f'{i}_f1', f'{i}_f2', f'{i}_f3', f'{i}_f4', f'{i}_f5'] for i in list(chanels)]).reshape(
        -1)
    data = pd.DataFrame(data=None, columns=['name'] + data_features.tolist())
    data['target'] = None

    print(np.where(seizures_targets == 'chb03/chb03_17.edf', seizures_targets))
    # Так как задача вычислительно трудозатратная решено было ее распараллелить
    #with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
    #    p.map_async(features_extraction, all_targets, callback=end_extraction)
    #    p.close()
    #    p.join()
