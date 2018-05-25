import math
import os
import numpy as np
import random as rd
import matplotlib.pyplot as plt


def read_train_data(address):
    f1 = os.listdir(address)
    rd.shuffle(f1)
    for row in f1:
        if row.endswith('.txt'):
            # print add
            label = os.path.splitext(row)[0]
            if label[0] == 'A': ds.append([1, -1, -1, -1, -1, -1, -1])
            elif label[0] == 'B': ds.append([-1, 1, -1, -1, -1, -1, -1])
            elif label[0] == 'C':   ds.append([-1, -1, 1, -1, -1, -1, -1])
            elif label[0] == 'D':     ds.append([-1, -1, -1, 1, -1, -1, -1])
            elif label[0] == 'E':      ds.append([-1, -1, -1, -1, 1, -1, -1])
            elif label[0] == 'J':       ds.append([-1, -1, -1, -1, -1, 1, -1])
            elif label[0] == 'K':        ds.append([-1, -1, -1, -1, -1, -1, 1])

            file = open(address + '/' + row)
            temp = []
            for line in file:
                line = line.strip("\r\n")
                for i in range(len(line)):
                    if line[i] == '.':
                        temp.append(-1)
                    elif line[i] == '#':
                        temp.append(1)
            train_data.append(temp)


def read_test_data(address):
    for row in os.listdir(address):
        if row.endswith('.txt'):
            test_label.append(os.path.splitext(row)[0][0])
            file = open(address + '/' + row)
            temp = []
            for line in file:
                line = line.strip("\r\n")
                for i in range(len(line)):
                    if line[i] == '.' or line[i] == 'o':
                        temp.append(-1)
                    elif line[i] == '#' or line[i] == '@':
                        temp.append(1)
            test_data.append(temp)


def activation(y_in):
    return (1 / (1 + np.exp(-y_in)))


def derivation(y_in):
    return activation(y_in) * (1 - activation(y_in))


def test_threshold(y_in):
    if y_in > 0:
        return 1
    else:
        return -1


def perceptron(train_data):
    train_data = np.array(train_data)
    counter = 0
    for q in range(100):
        counter += 1

        for i in range(len(train_data)):
            f_y_in = []
            y_in = []
            f_y_in1 = []
            y_in1 = []
            delta = []

            for j in range(W.shape[1]):
                y_in.append(B[j] + np.dot(train_data[i], W[:, j]))
                f_y_in.append(activation(y_in[j]))

            for j in range(V.shape[1]):
                y_in1.append(B1[j] + np.dot(f_y_in, V[:, j]))
                f_y_in1.append(activation(y_in1[j]))
                delta.append((ds[i][j] - f_y_in1[j]) * derivation(y_in1[j]))

            for j in range(W.shape[1]):
                delta = np.array(delta)
                delta_in = np.dot(delta, V[j].T)
                deltaa = delta_in * derivation(y_in[j])
                W[:, j] += alpha * deltaa * train_data[i].T
                B[j] = alpha * deltaa

            for j in range(V.shape[1]):
                f_y_in = np.array(f_y_in)
                V[:, j] += alpha * delta[j] * f_y_in.T
                B1[j] += alpha * delta[j]
    print('counter = ', counter)


def test_train(train_data):
    train_data = np.array(train_data)
    error = 0.0
    for i in range(train_data.__len__()):
        y_in = []
        f_y_in = []
        y_in1 = []
        f_y_in1 = []
        for j in range(W.shape[1]):
            y_in.append(B[j] + np.dot(train_data[i], W[:, j]))
            f_y_in.append(activation(y_in[j]))

        for j in range(V.shape[1]):
            y_in1.append(B1[j] + np.dot(f_y_in, V[:, j]))
            f_y_in1.append(test_threshold(activation(y_in1[j])))

        if f_y_in1 != ds[i]:
            error += 1
    print('Train Error = {}'.format(error / train_data.__len__() * 100))
    train_erroe.append(error / train_data.__len__() * 100)


def test_test(train_data):
    train_data = np.array(train_data)
    error = 0.0
    for i in range(train_data.__len__()):

        y_in = []
        f_y_in = []
        y_in1 = []
        f_y_in1 = []
        for j in range(W.shape[1]):
            y_in.append(B[j] + np.dot(train_data[i], W[:, j]))
            f_y_in.append(activation(y_in[j]))

        for j in range(V.shape[1]):
            y_in1.append(B1[j] + np.dot(f_y_in, V[:, j]))
            f_y_in1.append(test_threshold(activation(y_in1[j])))

        # print f_y_in
        if test_label[i] == 'A' and f_y_in1 != [1, -1, -1, -1, -1, -1, -1]:
            error += 1
        elif test_label[i] == 'B' and f_y_in1 != [-1, 1, -1, -1, -1, -1, -1]:
            error += 1
        elif test_label[i] == 'C' and f_y_in1 != [-1, -1, 1, -1, -1, -1, -1]:
            error += 1
        elif test_label[i] == 'D' and f_y_in1 != [-1, -1, -1, 1, -1, -1, -1]:
            error += 1
        elif test_label[i] == 'E' and f_y_in1 != [-1, -1, -1, -1, 1, -1, -1]:
            error += 1
        elif test_label[i] == 'J' and f_y_in1 != [-1, -1, -1, -1, -1, 1, -1]:
            error += 1
        elif test_label[i] == 'K' and f_y_in1 != [-1, -1, -1, -1, -1, -1, 1]:
            error += 1
    test_error.append(error / train_data.__len__() * 100)
    print('Test Error = {}'.format(error / train_data.__len__() * 100))


if __name__ == '__main__':
    alpha = 0.5
    train_data = []
    ds = []
    test_data = []
    test_label = []
    train_address = 'Characters-TrainSetHW4'
    test_address = 'Characters-TestSetHW4'
    read_train_data(train_address)
    read_test_data(test_address)
    train_erroe = [];
    test_error = [];
    W = (np.random.rand(63, 20) - 0.5) / 10
    V = (np.random.rand(20, 7) - 0.5) / 10
    B = (np.random.rand(20) - 0.5) / 10
    B1 = (np.random.rand(7) - 0.5) / 10
    perceptron(train_data)
    test_train(train_data)
    test_test(test_data)
