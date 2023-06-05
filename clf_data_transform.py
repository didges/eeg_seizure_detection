import numpy as np
import pandas as pd
import json
import pyedflib
import multiprocessing
import os
import pywt
import pyeeg
import scipy
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", UserWarning)


FREQUENCY = 256
NUM_FEATURES_FOR_CHANEL = 54
WINDOW_SIZE = 5
NON_SEIZURE_WINDOW = 8
CHANELS = ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4',
           'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2', 'FZ-CZ', 'CZ-PZ', 'P7-T7', 'T7-FT9', 'FT9-FT10', 'FT10-T8']
SEIZURE_JSON_PATH = 'seizures_time.json'


def metrics(signal):
    features = np.array([])
    coeffs = pywt.wavedec(signal, "coif3", level=7)
    cA7, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs
    for i in range(signal.shape[0]):
        for c_level in [cD1, cD2, cD3, cD4, cD5, cD6, cD7]:
            features = np.append(features, np.sum(c_level[i] ** 2))
            features = np.append(features, np.max(c_level[i]))
            features = np.append(features, np.min(c_level[i]))
            features = np.append(features, np.mean(c_level[i]))
            features = np.append(features, np.std(c_level[i]))
            features = np.append(features, scipy.stats.skew(c_level[i]))
            features = np.append(features, scipy.stats.kurtosis(c_level[i]))

        Kmax = 5
        Tau = 4
        DE = 10
        DFA = pyeeg.dfa(signal[i])
        HFD = pyeeg.hfd(signal[i], Kmax)
        SVD_Entropy = pyeeg.svd_entropy(signal[i], Tau, DE)
        Fisher_Information = pyeeg.fisher_info(signal[i], Tau, DE)
        PFD = pyeeg.pfd(signal[i])

        features = np.append(features, [DFA, HFD, SVD_Entropy, Fisher_Information, PFD])

    return features


def select_wind(sample, end):
    seizure_data = json.load(open(SEIZURE_JSON_PATH))
    is_seizure = np.array([], dtype=int)
    if sample in seizure_data.keys():
        selected_windows = np.array([], dtype=int)
        seiz_windows = np.array(seizure_data[sample]['start_time'])
        seiz_windows_end = np.array(seizure_data[sample]['end_time'])
        for cand in np.random.randint(0, end-FREQUENCY*WINDOW_SIZE, NON_SEIZURE_WINDOW):
            fits = True
            for s in range(len(seiz_windows)):
                if seiz_windows_end[s] > cand > seiz_windows[s]:
                    fits = False
            if fits:
                selected_windows = np.append(selected_windows, cand)
                is_seizure = np.append(is_seizure, 0)

        for seiz_start, seiz_end in zip(seiz_windows, seiz_windows_end):
            selected_windows = np.append(selected_windows, [seiz_start + i*WINDOW_SIZE for i in
                                                            range((seiz_end - seiz_start) // WINDOW_SIZE)])
            is_seizure = np.append(is_seizure, [1 for _ in range((seiz_end - seiz_start) // WINDOW_SIZE)])
    else:
        selected_windows = np.random.randint(0, end-FREQUENCY*WINDOW_SIZE, NON_SEIZURE_WINDOW)
        is_seizure = np.append(is_seizure, [0 for _ in range(NON_SEIZURE_WINDOW)])

    return selected_windows, is_seizure


def read_sig(sample_path):
    data_sample = pyedflib.EdfReader(sample_path)
    labels = data_sample.getSignalLabels()
    sample_channels = dict()
    if set(labels).intersection(set(CHANELS)) == set(CHANELS):
        for i, label in enumerate(labels):
            if label in set(CHANELS):
                sample_channels[label] = data_sample.readSignal(i)
        len_ch = len(sample_channels[CHANELS[0]])

        data = np.empty((22, len_ch))
        for i, ch in enumerate(CHANELS):
            data[i] = np.array(sample_channels[ch])

        data_sample.close()
        return data, len_ch
    return 0, 0


def features_extraction(sample):
    try:
        np.random.seed(int(os.getpid()*14))
        sample_channels, len_sig = read_sig(f"{dataset_path}/{sample}")
        if len(sample_channels):
            selected_windows, is_seizure = select_wind(sample, len_sig)
            data_features = np.array([[f'f{i}'] for i in range(NUM_FEATURES_FOR_CHANEL * 22)]).reshape(-1)
            data = pd.DataFrame(data=None, columns=['name'] + data_features.tolist())
            data['target'] = None
            for sel, is_seiz in zip(selected_windows, is_seizure):
                indices = np.arange(sel, sel + WINDOW_SIZE*FREQUENCY, dtype=int)
                sample_features = {'name': f"{sample}_{sel}"}

                features = metrics(sample_channels[:, indices])
                for i, feature in enumerate(features):
                    sample_features[f'f{i}'] = feature

                sample_features['target'] = is_seiz

                print(sample_features['name'])
                data.loc[len(data.index)] = sample_features

            data.to_csv(f'data/{sample.split("/")[1]}.csv')

    except Exception as e:
        print(f"EXCEPTION: {e}")


if __name__ == '__main__':
    dataset_path = "/home/didges/ds/eeg_project/EEG/physionet.org/files/chbmit/1.0.0"
    all_samples = open(f"{dataset_path}/RECORDS", "r").read().split("\n")[:-1]

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()-1) as p:
        p.map_async(features_extraction, all_samples)
        p.close()
        p.join()
