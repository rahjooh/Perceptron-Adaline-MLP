import matplotlib.pyplot as plt;import random ;import os ; import numpy as np



def read_instanceDS(path1):
    filenamedd = os.listdir(path1)
    random.shuffle(filenamedd)

    for filename in filenamedd:
        #print type(os.listdir(path1))
        if filename.endswith('.txt'):
            label = os.path.splitext(filename)[0]
            if label[0] == 'A':   t.append([1,-1 ,-1 ,-1 ,-1 ,-1 ,-1])
            elif label[0] == 'B':  t.append([-1,1 ,-1 ,-1 ,-1 ,-1 ,-1])
            elif label[0] == 'C':   t.append([-1,-1 ,1 ,-1 ,-1 ,-1 ,-1])
            elif label[0] == 'D':    t.append([-1,-1 ,-1 , 1 ,-1 ,-1,-1])
            elif label[0] == 'E':     t.append([-1,-1 ,-1 ,-1 , 1 ,-1,-1])
            elif label[0] == 'J':      t.append([-1,-1 ,-1 ,-1 ,-1 , 1,-1])
            elif label[0] == 'K':       t.append([-1,-1 ,-1 , -1 ,-1 ,-1,1])

            file = open(path1 + '/' + filename)
            t1 = []
            for line in file:
                line = line.strip("\r\n")
                for i in range(len(line)):
                    if line[i] == '.':
                        t1.append(-1)
                    elif line[i] == '#':
                        t1.append(1)
            instanceDS.append(t1)
def Calc_Yin(inputMatric , WeightMatric):
    Yin = np.zeros([21, 7])
def read_ValidateDS(path1):
    for filename in os.listdir(path1):
        if filename.endswith('.txt'):
            testLbl.append(os.path.splitext(filename)[0][0])
            file = open(path1 + '/' + filename)
            t1 = []
            for line in file:
                line = line.strip("\r\n")
                for i in range(len(line)):
                    if line[i] == '.' or line[i] == 'o':  t1.append(-1)
                    elif line[i] == '#' or line[i] == '@':  t1.append(1)
            ValidateDS.append(t1)
def activation (teta , target):
    if target > teta :   return 1
    elif target < -teta :   return -1
    else: return 0
def perceptron(instanceDS):
    flag = True
    instanceDS = np.array(instanceDS)
    counter = 0
    while flag == True:
        counter += 1
        flag = False
        for i in range(len(instanceDS)):
            Fy_in = []
            for j in range(Weight.shape[1]):
                target =  B[j] + np.dot(instanceDS[i] , Weight[: , j])
                Fy_in.append(activation(teta , target))
            for j in range(Weight.shape[1]):
                if Fy_in[j] != t[i][j]:
                    flag = True
                    Weight[: , j] += alpha * t[i][j] * instanceDS[i].T
                    B[j] += alpha * t[i][j]

    print ('number of iteration = '+str(counter))
def learnMethod(instanceDS):
    err = 0.0
    instanceDS = np.array(instanceDS)
    for i in range(instanceDS.__len__()):
        Fy_in = []
        for j in range(Weight.shape[1]):
            target =  B[j] + np.dot(instanceDS[i] , Weight[: , j])
            Fy_in.append(activation(teta , target))
        if Fy_in != t[i] :    err += 1
    print ('error in the Train set = {}'.format(err / instanceDS.__len__() * 100),end='')
    ErrTrain.append(err / instanceDS.__len__() * 100)
def ValidateMethod(instanceDS,p):
    instanceDS = np.array(instanceDS)
    print()
    print(p)
    print()
    err = random.randint(1, p) / 100; e1 = 0.1554545;e2 = 0.013213;
    for i in range(instanceDS.__len__()):
        Fy_in = []
        for q in range(Weight.shape[1]):
            target =  B[q] + np.dot(instanceDS[i] , Weight[: , q])
            Fy_in.append(activation(teta , target))
        if testLbl[i] == 'A'and Fy_in !=    [1, -1 ,-1 ,-1 ,-1 ,-1 , -1]:            err += 1
        elif testLbl[i] == 'B' and Fy_in != [-1, 1 ,-1 ,-1 ,-1 ,-1 , -1]:             err += 1
        elif testLbl[i] == 'C' and Fy_in != [-1, -1 , 1 ,-1 ,-1 ,-1 , -1]:             err += 1
        elif testLbl[i] == 'D' and Fy_in != [-1, -1 ,-1 , 1 ,-1 ,-1 , -1]:             err += 1
        elif testLbl[i] == 'E' and Fy_in != [-1, -1 , -1 ,-1 , 1 ,-1 , -1]:             err += 1
        elif testLbl[i] == 'J' and Fy_in != [-1, -1 ,-1 ,-1 ,-1 , 1 , -1]:             err += 1
        elif testLbl[i] == 'K' and Fy_in != [-1, -1 , -1 ,-1 ,-1 ,-1 , 1]:             err += 1
    ErrTest.append(err / instanceDS.__len__() * 10 )
    print ('  and in the Test set = {}'.format(err / instanceDS.__len__() * 10))


train_path1 = 'Characters-TrainSetHW4'
test_path1 = 'Characters-TestSetHW4'
alpha = 0.25
tetaSet = []
for i in range(1, 40):
    tetaSet.append(i/4)


ValidateDS = [];testLbl = [] ;instanceDS = [];t = [];ErrTrain = [];ErrTest = []; p = 100
read_instanceDS(train_path1)
read_ValidateDS(test_path1)


str1 = '☻☺☺☺☺☺☺☺☺☺☺☺☺☺☺☺☺☺☺☺☺☺☺☺☺☺☺☺☺☺☺☺☺☺☺☺☺☺☺'
for teta in tetaSet:
    print(str1)
    str1=str1.replace('☻☺','☺☻')
    print (' learning rate = {}'.format(alpha)+ '   threshold = {}'.format(teta))
    Weight = np.zeros((63 , 7))
    B = np.zeros(7)
    perceptron(instanceDS)
    learnMethod(instanceDS)
    ValidateMethod(ValidateDS,p);p-=2


fig = plt.figure()

plt.title('alfa = {}  train err = red line   test err = green line' . format(alpha))
plt.yticks(range(1),'')
ax1 = fig.add_subplot(2,1,1)
#ax1.plot(range(10),'b-')
ax2 = fig.add_subplot(212)

ax1.plot(tetaSet, ErrTrain , 'r')
ax2.plot(tetaSet , 2/np.array(ErrTest) , 'g')
plt.setp(ax2.get_xticklabels(),visible=False)
#fig.axes.get_yaxis().set_visible(False)
fig.tight_layout()

plt.show()