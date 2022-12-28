import numpy as np
import pandas as pd
import json
import multiprocessing
import os
from clf_data_transform import end_extraction, read_sig

FREQUENCY = 256
NUM_FEATURES_FOR_CHANEL = 31
WINDOW_SIZE = 30
CHANELS = ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4',
           'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2', 'FZ-CZ', 'CZ-PZ', 'P7-T7', 'T7-FT9', 'FT9-FT10', 'FT10-T8']
SEIZURE_JSON_PATH = 'seizures_time.json'
K = 16


def decimate_signal(signal, k):
    """
    Децимация сигнала на k
    """
    dec = list(range(0, len(signal), k))
    return np.array(signal)[dec]


def select_regr_wind(sample):
    """
    Функция выбирает окна, которые будут выбраны для обработки.
    Для каждого таргета выбирается по 3 окна со случайным смещением начала.
    Возвращает массив с началом выбранных окон, смещение
    """
    seizure_data = json.load(open(SEIZURE_JSON_PATH))
    start_time_in_window = np.array([])
    selected_windows = np.array([], dtype=int)
    if sample in seizure_data.keys():
        seiz_windows = np.array(seizure_data[sample]['start_time'])
        shift_windows = np.random.randint(0, 9), np.random.randint(10, 19), np.random.randint(20, 25)
        for shift_sample in seiz_windows:
            for shift_wind in shift_windows:
                selected_windows = np.append(selected_windows, (shift_sample-shift_wind)*256)
                start_time_in_window = np.append(start_time_in_window, shift_wind)

    return selected_windows, start_time_in_window


def regr_extraction(sample):
    try:
        np.random.seed(int(os.getpid()*14))
        sample_channels, _ = read_sig(f"{dataset_path}/{sample}")
        if len(sample_channels):
            selected_windows, shift = select_regr_wind(sample)
            result = []
            for sel, sh in zip(selected_windows, shift):
                indices = np.arange(sel, sel + WINDOW_SIZE*FREQUENCY, dtype=int)
                features = np.array([])
                sample_features = {'name': f"{sample}_{sel}"}
                for ch in CHANELS:
                    features = np.append(features, decimate_signal(sample_channels[ch][indices], K), axis=0)
                for i, feature in enumerate(features.tolist()):
                    sample_features[f'f{i}'] = feature

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
    all_samples = open(f"{dataset_path}/RECORDS-WITH-SEIZURES", "r").read().split("\n")[:-1]
    data_features = np.array([[f'f{i}'] for i in range(int(22*30*FREQUENCY/K))]).reshape(-1)
    data = pd.DataFrame(data=None, columns=['name'] + data_features.tolist())
    data['shift'] = None

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()-1) as p:
        p.map_async(regr_extraction, all_samples, callback=end_extraction)
        p.close()
        p.join()
