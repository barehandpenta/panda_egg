import random as rd
import numpy as np
import csv

data_path = '../data/'
# Shorter code for loading CSV file:
def open_csv_file(name, type):
    if type == 'r':
        data = open(data_path + name, 'r').readlines()
        for i in range(len(data)):
            data[i] = np.asfarray(data[i].split(','))
    else:
        data = open(data_path + name, 'w', newline='')
        data = csv.writer(data)
    return data
# Shuffle from random librady doesn't return a value, so i rewrite a little bit:
def my_shuffle(array):
    rd.shuffle(array)
    return array
# Function to copy data from array1 and paste  to array2
def copy_paste(arr1, arr2):
    for i in arr1:
        arr2.append(i)

def preProcessing(filename):
    train_examples = open_csv_file(filename, 'r')
    inputs = [None]*len(train_examples)
    targets = [None]*len(train_examples)
    for i in range(len(train_examples)):
        inputs[i] = np.asfarray(train_examples[i][1:])
        targets[i] = np.zeros(3) + 0.01
        targets[i][int(train_examples[i][0])] = 0.99
    return inputs, targets