import numpy as np
import pandas as pd
import json
import pyedflib
import multiprocessing
import pyeeg

FREQUENCY = 256
NUM_FEATURES_FOR_CHANEL = 31


def metrics(mat):
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

    feature_pandas = pd.DataFrame(np.array(mat))
    data_transform = feature_pandas.describe().iloc[1:, :]

    # Коэффициент асимметрии
    feature_pandas.loc['skew'] = feature_pandas.skew().tolist()

    # Среднее абсолютное отклонение
    data_transform.loc['mad'] = feature_pandas.mad().tolist()

    # Коэффициент эксцесса — мера остроты пика распределения случайной величины.
    data_transform.loc['kurtosis'] = feature_pandas.kurtosis().tolist()

    # добавление квантилей
    for i in range(0, 100, 5):
        if (i != 25) & (i != 50) & (i != 75):
            str_col = f"{i}%"
            int_col = float(i) / 100
            data_transform.loc[str_col] = feature_pandas.quantile(int_col).tolist()
        else:
            continue

    pd_f = data_transform.T.values[0].tolist()

    return (DFA, HFD, SVD_Entropy, Fisher_Information, PFD) + tuple(pd_f)


def window_border(len_signal, wind_size=FREQUENCY * 30):
    indices = np.arange(len_signal)
    for start in range(0, len_signal, wind_size):
        end = min(start + wind_size, len_signal)
        window_idx = indices[start:end]
        yield window_idx


def end_extraction(response):
    counter = 0
    try:
        for r in response:
            for el in r:
                counter += 1
                print(counter)
                data.loc[len(data.index)] = el
        print(f"{counter} observations will be write")
        data.to_csv('dataset_window_31metrics.csv')
    except Exception as e:
        print(f"EXCEPTION:{e}")


def features_extraction(sample):
    try:
        data_sample = pyedflib.EdfReader(f"{dataset_path}/{sample}")
        labels = data_sample.getSignalLabels()
        sample_channels = dict()
        if set(labels).intersection(set(chanels)) == set(chanels):
            for i, label in enumerate(labels):
                if label in set(chanels):
                    sample_channels[label] = data_sample.readSignal(i)
            selected_windows = np.random.randint(0, high=len(sample_channels[chanels[0]]) - 30 * 256, size=5)
            if sample in seizures_targets:
                selected_windows = np.append(selected_windows, np.array(seizure_data[sample]['start_time']) * FREQUENCY, axis=0)
            selected_windows.sort()
            ret = []
            for sel in selected_windows:
                for window_index in window_border(len(sample_channels[chanels[0]])):
                    if window_index[0] <= sel <= window_index[-1]:
                        features = np.array([])
                        sample_features = {'name': f"{sample}_{window_index[0]}"}
                        for ch in chanels:
                            features = np.append(features, metrics(sample_channels[ch][window_index]), axis=0)
                        for i, v in enumerate(features.tolist()):
                            sample_features[f'v{i}'] = v

                        if sample in seizures_targets:
                            for seiz in seizure_data[sample]['start_time']:
                                if window_index[0] / FREQUENCY <= seiz <= window_index[-1] / FREQUENCY:
                                    sample_features['target'] = 1
                                else:
                                    sample_features['target'] = 0
                        else:
                            sample_features['target'] = 0

                        print(sample_features['name'])
                        ret.append(sample_features)
            return ret
        else:
            return []
    except Exception as e:
        print(f"EXCEPTION: {e}")


if __name__ == '__main__':
    dataset_path = "../EEG/physionet.org/files/chbmit/1.0.0/"

    chanels = ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4',
               'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2', 'FZ-CZ', 'CZ-PZ', 'P7-T7', 'T7-FT9', 'FT9-FT10', 'FT10-T8']

    all_samples = open(f"{dataset_path}/RECORDS", "r").read().split("\n")[:-1]
    seizure_data = json.load(open("seizures_time.json"))
    seizures_targets = seizure_data.keys()
    data_features = np.array([[f'v{i}'] for i in range(NUM_FEATURES_FOR_CHANEL*22)]).reshape(-1)
    data = pd.DataFrame(data=None, columns=['name'] + data_features.tolist())
    data['target'] = None

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()-1) as p:
        p.map_async(features_extraction, all_samples, callback=end_extraction)
        p.close()
        p.join()

