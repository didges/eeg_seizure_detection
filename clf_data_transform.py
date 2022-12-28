import numpy as np
import pandas as pd
import json
import pyedflib
import multiprocessing
import pyeeg
import os


FREQUENCY = 256
NUM_FEATURES_FOR_CHANEL = 31
WINDOW_SIZE = 30
CHANELS = ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4',
           'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2', 'FZ-CZ', 'CZ-PZ', 'P7-T7', 'T7-FT9', 'FT9-FT10', 'FT10-T8']
SEIZURE_JSON_PATH = 'seizures_time.json'


def metrics(mat):
    """
    Функция принимает на вход 1d сигнал и возвращает tuple из 31 метрики
    """
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


def end_extraction(response, name="dataset_shiftwindow_31metrics.csv"):
    """
    Завершающий callback для параллельного подсчета, записывает данные в csv файл
    """
    counter = 0
    try:
        for r in response:
            for el in r:
                counter += 1
                data.loc[len(data.index)] = el
        print(f"{counter} observations will be write")
        data.to_csv(name)
    except Exception as e:
        print(f"EXCEPTION:{e}")


def select_wind(sample, end):
    """
    Функция выбирает окна, которые будут выбраны для обработки.
    Всегда выбирается 5 случайных окон в которых нет таргета.
    Для каждого таргета выбирается по 3 окна со случайным смещением начала.
    Возвращает массив с началом выбранных окон, массив является ли окно таргетом, смещение
    """
    seizure_data = json.load(open(SEIZURE_JSON_PATH))
    is_seizure = np.array([], dtype=int)
    start_time_in_window = np.array([])
    if sample in seizure_data.keys():
        selected_windows = np.array([], dtype=int)
        seiz_windows = np.array(seizure_data[sample]['start_time'])
        seiz_windows_end = np.array(seizure_data[sample]['end_time'])
        rand_added = 0
        while rand_added < 5:
            candidate = np.random.randint(0, high=end-WINDOW_SIZE*FREQUENCY)
            fits = True
            for s in range(len(seiz_windows)):
                if seiz_windows_end[s] > s > seiz_windows[s]:
                    fits = False
            if fits:
                selected_windows = np.append(selected_windows, candidate)
                is_seizure = np.append(is_seizure, 0)
                rand_added += 1
                start_time_in_window = np.append(start_time_in_window, None)

        shift_windows = np.random.randint(0, 9), np.random.randint(10, 19), np.random.randint(20, 25)
        for shift_sample in seiz_windows:
            for shift_wind in shift_windows:
                selected_windows = np.append(selected_windows, (shift_sample-shift_wind)*256)
                start_time_in_window = np.append(start_time_in_window, shift_wind)

        is_seizure = np.append(is_seizure, [1 for _ in range(len(seiz_windows)*3)])
    else:
        selected_windows = np.random.randint(0, high=end-WINDOW_SIZE*FREQUENCY, size=5, dtype=int)
        is_seizure = np.append(is_seizure, [0 for _ in range(len(selected_windows))])
        start_time_in_window = np.append(start_time_in_window, [None for _ in range(len(selected_windows))])

    return selected_windows, is_seizure, start_time_in_window


def read_sig(sample_path):
    data_sample = pyedflib.EdfReader(sample_path)
    labels = data_sample.getSignalLabels()
    sample_channels = dict()
    if set(labels).intersection(set(CHANELS)) == set(CHANELS):
        for i, label in enumerate(labels):
            if label in set(CHANELS):
                sample_channels[label] = data_sample.readSignal(i)

    return sample_channels, len(sample_channels[CHANELS[0]])


def features_extraction(sample):
    try:
        np.random.seed(int(os.getpid()*14))
        sample_channels, len_sig = read_sig(f"{dataset_path}/{sample}")
        if len(sample_channels):
            selected_windows, is_seizure, shift = select_wind(sample, len_sig)
            result = []
            for sel, is_seiz, sh in zip(selected_windows, is_seizure, shift):
                indices = np.arange(sel, sel + WINDOW_SIZE*FREQUENCY, dtype=int)
                features = np.array([])
                sample_features = {'name': f"{sample}_{sel}"}
                for ch in CHANELS:
                    features = np.append(features, metrics(sample_channels[ch][indices]), axis=0)
                for i, feature in enumerate(features.tolist()):
                    sample_features[f'f{i}'] = feature

                sample_features['target'] = is_seiz
                sample_features['shift'] = sh

                print(sample_features['name'])
                result.append(sample_features)
            return result
        else:
            return []
    except Exception as e:
        print(f"EXCEPTION: {e}")


if __name__ == '__main__':
    dataset_path = "../EEG/physionet.org/files/chbmit/1.0.0/"
    all_samples = open(f"{dataset_path}/RECORDS", "r").read().split("\n")[:-1]
    data_features = np.array([[f'f{i}'] for i in range(NUM_FEATURES_FOR_CHANEL*22)]).reshape(-1)
    data = pd.DataFrame(data=None, columns=['name'] + data_features.tolist())
    data['target'] = None
    data['shift'] = None

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()-1) as p:
        p.map_async(features_extraction, all_samples, callback=end_extraction)
        p.close()
        p.join()
    #sample_channels = read_sig(f"{dataset_path}/{all_samples[2]}")
    #print(select_wind(all_samples[2], len(sample_channels[CHANELS[0]])))