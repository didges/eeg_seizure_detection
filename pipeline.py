from catboost import CatBoostClassifier, CatBoostRegressor, Pool
import numpy as np
from clf_data_transform import read_sig, metrics
from reg_data_transform import decimate_signal

FREQUENCY = 256
WINDOW_SIZE = 30
CHANELS = ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4',
           'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2', 'FZ-CZ', 'CZ-PZ', 'P7-T7', 'T7-FT9', 'FT9-FT10', 'FT10-T8']


class SeizureModel:
    def __init__(self, clf_model, reg_model):
        self.clf_model = CatBoostClassifier().load_model(clf_model)
        self.reg_model = CatBoostRegressor().load_model(reg_model)
        self.overlap = 0
        self.sig = None
        self.len_sig = None
        self.K = 16

    def set_overlap(self, overlap):
        if 30 > overlap > 0:
            self.overlap = overlap
        else:
            print("Overlap must be between 0 and 30")

    def __select_windows(self):
        extracted_windows = []
        start_time = []
        windows_start = np.arange(0, (self.len_sig-WINDOW_SIZE*FREQUENCY)//FREQUENCY, WINDOW_SIZE-self.overlap)
        windows_start = np.append(windows_start, self.len_sig//FREQUENCY-WINDOW_SIZE)
        for start_window_time in windows_start:
            indices = np.arange(start_window_time*FREQUENCY, (start_window_time + WINDOW_SIZE)*FREQUENCY, dtype=int)
            extract = np.array([])
            for ch in CHANELS:
                extract = np.append(extract, metrics(self.sig[ch][indices]), axis=0)
            extracted_windows.append(extract)
            start_time.append(start_window_time*FREQUENCY)
        return extracted_windows, start_time

    def __detect_windows(self):
        detected = []
        windows, start_time = self.__select_windows()
        clf_pred = self.clf_model.predict(Pool(windows))
        for det in np.where(np.array(clf_pred) == 1)[0]:
            detected.append(start_time[det])

        return detected

    def __detect_start_time(self, windows_start):

        detected_start_time = []
        for w in windows_start:
            data = np.array([])
            indices = np.arange(w, w + WINDOW_SIZE * FREQUENCY, dtype=int)
            for ch in CHANELS:
                data = np.append(data, decimate_signal(self.sig[ch][indices], self.K), axis=0)
            for ch in CHANELS:
                data = np.append(data, metrics(self.sig[ch][indices]), axis=0)    
            
            detected_start_time.append(w//FREQUENCY + int(self.reg_model.predict(data)))

        return detected_start_time

    def predict(self, sample):
        self.sig, self.len_sig = read_sig(sample)
        detected_windows = self.__detect_windows()
        if detected_windows:
            detected_start_time = self.__detect_start_time(detected_windows)
            for s in detected_start_time:
                print(f"Обнаружен всплеск на {s} секунде")
            return detected_start_time
        else:
            print(f"Всплесков не обнаружено")
            return []


if __name__ == '__main__':
    s = "/home/didges/ds_learning/EEG/physionet.org/files/chbmit/1.0.0/chb04/chb04_28.edf"
    model = SeizureModel('window_detection.cbm', 'start_time.cbm')
    #model.set_overlap(15)
    model.predict(s)

