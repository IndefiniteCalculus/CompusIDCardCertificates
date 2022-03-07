import os
from CharacterIdentification import ConfigReader as conf
import csv
import numpy as np
def get_labels():
    labels = []
    dir, _ = conf.get_dir_Chinese_Characters()
    dir += "\\labels.txt"
    with open(dir, encoding='UTF-8') as file:
        csv_reader = csv.reader(file)
        ascii_label = [int('0x4e00',16)]
        error_label = []
        for row in csv_reader:
            if len(row) == 0:
                continue
            start_with = row[0]
            if len(start_with) == 1:
                if start_with == '#':
                    continue
            else:
                if start_with[0] == '#':
                    continue
            for label in row:
                labels.append(chr(int(label,16)))
                # ascii_label = [int(label, 16)] + ascii_label
                # d = ascii_label[1] - ascii_label[0]
                # ascii_label.pop()
                # if d >= 1 or d == 0:
                #     error_label.append(label)
        labels.sort()
        pass
    return labels
def encoded_onehot_label(length):
    m = np.eye(length,length)
    return m
def encoded_idx_label(length):
    return [i for i in range(length)]

def decode_label(label:np.array):
    labels = get_labels()
    max_idx = 0
    for i in range(len(label)):
        if label[i] > label[max_idx]:
            max_idx = i
    return labels[max_idx]
