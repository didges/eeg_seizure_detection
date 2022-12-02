import os
import json

if __name__ == "__main__":
    data = {}
    dataset_path = "../EEG/physionet.org/files/chbmit/1.0.0/"

    all_targets = open(f"{dataset_path}/RECORDS", "r").read().split("\n")[:-1]
    seizures_targets = open(f"{dataset_path}/RECORDS-WITH-SEIZURES", "r").read().split("\n")[:-1]

    subjects = [f'{dataset_path}/{i}' for i in os.listdir(dataset_path) if os.path.isdir(f'{dataset_path}/{i}')]
    sums = sorted([f'{i}/{i.rsplit("/")[-1]}-summary.txt' for i in subjects])

    for seizure in seizures_targets:
        subject_num = int(seizure.split("/")[0][3:])
        file = open(sums[subject_num-1]).read()
        sample_summary = file.split(seizure.split("/")[1])[1].split("File Name:")[0]
        n_seizures = int(sample_summary.split("Number of Seizures in File:")[1].split('\n')[0])
        if n_seizures == 1:
            start_time = int(sample_summary.split("Start Time:")[-1].split('seconds')[0])
            end_time = int(sample_summary.split("End Time:")[-1].split('seconds')[0])
            data[seizure] = {"start_time": start_time, "end_time": end_time}
        elif n_seizures > 1:
            start_time_arr, end_time_arr = [], []
            for i in range(1, n_seizures+1):
                start_time_arr.append(int(sample_summary.split(f"Seizure {i} Start Time:")[1].split('seconds')[0]))
                end_time_arr.append(int(sample_summary.split(f"Seizure {i} End Time:")[1].split('seconds')[0]))
            data[seizure] = {"start_time": start_time_arr, "end_time": end_time_arr}

        with open("seizures_time.json", 'w') as fp:
            json.dump(data, fp)