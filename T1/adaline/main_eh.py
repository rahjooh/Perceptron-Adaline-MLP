import os
import numpy as np
import random
import matplotlib.pyplot as plt

__author__ = 'EHSAN'

def read_train_data(address):
    adddd = os.listdir(address)
    random.shuffle(adddd)

    for add in adddd:
        #print type(os.listdir(address))
        if add.endswith('.txt'):
            print (add)
            label = os.path.splitext(add)[0]
            if label[0] == 'A':
                t.append([1, -1 ,-1 ,-1 ,-1 ,-1 , -1])
            elif label[0] == 'B':
                t.append([-1, 1 ,-1 ,-1 ,-1 ,-1 , -1])
            elif label[0] == 'C':
                t.append([-1, -1 , 1 ,-1 ,-1 ,-1 , -1])
            elif label[0] == 'D':
                t.append([-1, -1 , -1 , 1 ,-1 ,-1 , -1])
            elif label[0] == 'E':
                t.append([-1, -1 ,-1 ,-1 , 1 ,-1 , -1])
            elif label[0] == 'J':
                t.append([-1, -1 , -1 ,-1 ,-1 , 1 , -1])
            elif label[0] == 'K':
                t.append([-1, -1 , -1 , -1 ,-1 ,-1 , 1])

            file = open(address + '/' + add)
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
    for add in os.listdir(address):
        if add.endswith('.txt'):
            test_label.append(os.path.splitext(add)[0][0])
            file = open(address + '/' + add)
            temp = []
            for line in file:
                line = line.strip("\r\n")
                for i in range(len(line)):
                    if line[i] == '.' or line[i] == 'o':
                        temp.append(-1)
                    elif line[i] == '#' or line[i] == '@':
                        temp.append(1)
            test_data.append(temp)


def activation (theta , y_in):
    if y_in > theta :
        return 1
    elif y_in < -theta :
        return -1
    else:
        return 0

def perceptron(train_data):
    flag = True
    train_data = np.array(train_data)
    counter = 0
    while flag == True:
        counter += 1
        flag = False
        for i in range(len(train_data)):
            f_y_in = []
            for j in range(W.shape[1]):
                y_in =  B[j] + np.dot(train_data[i] , W[: , j])
                f_y_in.append(activation(theta , y_in))
            for j in range(W.shape[1]):
                if f_y_in[j] != t[i][j]:
                    flag = True
                    W[: , j] += alpha * t[i][j] * train_data[i].T
                    B[j] += alpha * t[i][j]
            #print W
        print (counter)

def test_train(train_data):
    train_data = np.array(train_data)
    error = 0.0
    for i in range(train_data.__len__()):
        f_y_in = []
        for j in range(W.shape[1]):
            y_in =  B[j] + np.dot(train_data[i] , W[: , j])
            f_y_in.append(activation(theta , y_in))
        if f_y_in != t[i] :
            error += 1
    print ('Train Error = {}'.format(error / train_data.__len__() * 100))
    train_erroe.append(error / train_data.__len__() * 100)

def test_test(train_data):
    train_data = np.array(train_data)
    error = 0.0
    for i in range(train_data.__len__()):
        f_y_in = []
        for j in range(W.shape[1]):
            y_in =  B[j] + np.dot(train_data[i] , W[: , j])
            f_y_in.append(activation(theta , y_in))

        '''
        for k in range(len(f_y_in)):
            if f_y_in[k] == 0 and 1 in f_y_in:
                f_y_in[k] = -1
            if f_y_in[k] == 0 and 1 not in f_y_in:
                f_y_in[k] = 1
        '''
        #print f_y_in
        if test_label[i] == 'A'and f_y_in != [1, -1 ,-1 ,-1 ,-1 ,-1 , -1]:
            error += 1
        elif test_label[i] == 'B' and f_y_in != [-1, 1 ,-1 ,-1 ,-1 ,-1 , -1]:
            error += 1
        elif test_label[i] == 'C' and f_y_in != [-1, -1 , 1 ,-1 ,-1 ,-1 , -1]:
            error += 1
        elif test_label[i] == 'D' and f_y_in != [-1, -1 ,-1 , 1 ,-1 ,-1 , -1]:
            error += 1
        elif test_label[i] == 'E' and f_y_in != [-1, -1 , -1 ,-1 , 1 ,-1 , -1]:
            error += 1
        elif test_label[i] == 'J' and f_y_in != [-1, -1 ,-1 ,-1 ,-1 , 1 , -1]:
            error += 1
        elif test_label[i] == 'K' and f_y_in != [-1, -1 , -1 ,-1 ,-1 ,-1 , 1]:
            error += 1
    test_error.append(error / train_data.__len__() * 100 )
    print ('Test Error = {}'.format(error / train_data.__len__() * 100))

if __name__ == '__main__':
    alpha = 0.5
    #theta = 2
    theta_set = [0.5 , 1 , 1.5 ,2 , 2.5 , 3 , 3.5 , 4 , 4.5 , 5]
    train_data = []
    t = []
    test_data = []
    test_label = []
    train_address = 'Characters-TrainSetHW4'
    test_address = 'Characters-TestSetHW4'
    read_train_data(train_address)
    read_test_data(test_address)
    train_erroe = []
    test_error = []
    for theta in theta_set:
        print ('')
        print ('theta = {}'.format(theta))
        W = np.zeros((63 , 7))
        B = np.zeros(7)
        perceptron(train_data)
        test_train(train_data)
        test_test(test_data)
    plt.plot(theta_set, train_erroe , 'r')
    plt.plot(theta_set , test_error , 'b')
    plt.xlabel('Theta')
    plt.ylabel('Error')
    plt.title('alpha = {}' . format(alpha))
    plt.show()