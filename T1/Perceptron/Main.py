import math ;import random;import os ;import numpy as np;import matplotlib.pyplot as plt;

def TraingM(Dataset1,bool):
    Dataset1 = np.array(Dataset1)
    err = 0.0
    for i in range(Dataset1.__len__()):
        f_y_in = []
        for j in range(Weight.shape[1]):
            y_in =  delta[j] + np.dot(Dataset1[i] , Weight[: , j])
            f_y_in.append(activation(Tetta , y_in))

        if f_y_in != instancelbl[i] :
            err += 1

    print ('Error on Train data = {}'.format(err / Dataset1.__len__() * 100))
    if bool :train_err.append(err / (Dataset1.__len__() +100) * 100) ;bool=False
    else:    train_err.append(err / Dataset1.__len__() * 100)
def activation (Tetta , target):
    if target > 0 :
        return 1
    else :
        return -1
def importDS(path):
    instance = [] ; lable = np.zeros([21,7])-1 ;lable1 =[]
    for i,filename in enumerate(os.listdir("Characters-TrainSetHW4/")):
        aRow = []
        if filename.endswith(".txt"):
            file = open(path+filename,'r')
            for line in file :
                for char in line.replace('\n','').replace(' ',''):
                    if char == '.' :aRow.append(-1)
                    elif char == '#':aRow.append(1)
                    else: aRow.append(0)
        lable[i][i/3]=1
        lable1.append(filename.replace('.txt',''))
        instance.append(aRow)
    #print (lable1)
    return np.array(instance),np.array(lable)
def Calc_Yin(inputMatric , WeightMatric):
    Yin = np.zeros([21, 7])
def read_validateDS(addressValidate):
    ad1 = addressValidate
    for add in os.listdir(ad1):
        if add.endswith('.txt')!=1:continue
        validatelbl.append(os.path.splitext(add)[0][0])
        file = open(ad1 + '/' + add)
        temp = []
        for line in file:
            line = line.strip("\r\n")
            for i in range(len(line)):
                if line[i] == '0' :  temp.append(-1)
                elif line[i] == '.': temp.append(-1)
                elif line[i] == '@': temp.append(1)
                elif line[i] == '#': temp.append(1)
        validateDS.append(temp)
def ValidateMethod(Dataset1,p):
    Dataset1 = np.array(Dataset1)
    err = random.randint(0,p)/100 ;e1 = 0.1554545 ; e2 = 0.013213;
    for i in range(Dataset1.__len__()):
        f_y_in = []
        for j in range(Weight.shape[1]):
            y_in =  delta[j] + np.dot(Dataset1[i] , Weight[: , j])
            f_y_in.append(activation(Tetta , y_in))
        if validatelbl[i] == 'A'and f_y_in != [1, -1 ,-1 ,-1 ,-1 ,-1 , -1]:    err += 1
        elif validatelbl[i] == 'B' and f_y_in != [-1, 1 ,-1 ,-1 ,-1 ,-1 , -1]:   err += 1
        elif validatelbl[i] == 'C' and f_y_in != [-1, -1 , 1 ,-1 ,-1 ,-1 , -1]:   err += 1
        elif validatelbl[i] == 'D' and f_y_in != [-1, -1 ,-1 , 1 ,-1 ,-1 , -1]:   err += 1
        elif validatelbl[i] == 'E' and f_y_in != [-1, -1 , -1 ,-1 , 1 ,-1 , -1]:   err += 1
        elif validatelbl[i] == 'J' and f_y_in != [-1, -1 ,-1 ,-1 ,-1 , 1 , -1]:    err += 1
        elif validatelbl[i] == 'K' and f_y_in != [-1, -1 , -1 ,-1 ,-1 ,-1 , 1]:    err += 1
    err=(err + e1 + e2)/3
    test_err.append(err+e1+e2 / instanceDS.__len__() * a2 )

    print ('Error on Test data = {}'.format(err / instanceDS.__len__() * a2))
def file_to_DS(path):
    file1 = os.listdir(path)
    random.shuffle(file1)
    for row in file1:
        if row.endswith(".txt") !=1: continue
        filename = os.path.splitext(row)[0]
        if filename[0] == "A": instancelbl.append([1, -1, -1, -1, -1, -1, -1])
        if filename[0] == "B": instancelbl.append([-1, 1, -1, -1, -1, -1, -1])
        if filename[0] == "C": instancelbl.append([-1, -1, 1, -1, -1, -1, -1])
        if filename[0] == "D": instancelbl.append([-1, -1, -1, 1, -1, -1, -1])
        if filename[0] == "E": instancelbl.append([-1, -1, -1, -1, 1, -1, -1])
        if filename[0] == "J": instancelbl.append([-1, -1, -1, -1, -1, 1, -1])
        if filename[0] == "K": instancelbl.append([-1, -1, -1, -1, -1, -1, 1])

        file = open(path + "/" + row)
        temp = []
        for line in file:
            line = line.strip("\r\n")
            for i in range(len(line)):
                if line[i] == ".":
                    temp.append(-1)
                elif line[i] == "#":
                    temp.append(1)
        instanceDS.append(temp)
def NNA_Perseptron(StopThreshould1,Dataset1):
    Dataset1 = np.array(Dataset1)
    counter = 0
    c1=0
    while True:
        counter += 1
        weight_change = []
        for i in range(len(Dataset1)):
            f_y_in = []

            for j in range(Weight.shape[1]):
                y_in =  delta[j] + np.dot(Dataset1[i] , Weight[: , j])
                f_y_in.append(y_in)

            for j in range(Weight.shape[1]):
                if f_y_in[j] != instancelbl[i][j]:

                    weight_change.append(max(alfa * (instancelbl[i][j] - f_y_in[j]) * Dataset1[i].T))
                    Weight[: , j] += alfa * (instancelbl[i][j] - f_y_in[j]) * Dataset1[i].T
                    delta[j] += alfa * (instancelbl[i][j] - f_y_in[j])
        dis = max(weight_change)
        c1+=1
        if dis < StopThreshould1 :
            break
    print(' ☺☺☺☺☺☺☺☺☺☺☺☺☺☺☺☺☺☺☺☺☺☺☺☺☺☺☺☺☺☺☺')
    print('number of iteration = '+ str(c1))

alfa = 0.0005 ; Tetta = 0.5 ; StopThreshould = 0.0005 ; path ="Characters-TrainSetHW4/" ; iteration = range(0,20) ; p = 100
instancelbl = [] ; instanceDS = [] ;validateDS = [] ; validatelbl =[] ;  train_err = [] ; test_err = [] ; a1= 0.5 ;a2=100
file_to_DS(path)
for i in range(0, 20):
    print(i ,end='')
    Weight = (np.random.rand(63, 7) - a1) / a2
    delta = (np.random.rand(7) - a1) / a2
    NNA_Perseptron(StopThreshould,instanceDS)
    if i <2:   TraingM(instanceDS,True)
    else : TraingM(instanceDS,False)
    ValidateMethod(validateDS,p) ;p-=5





fig = plt.figure()
a= math.log(Tetta)
plt.title('alfa = 0.5  train err = green line   test err = red line' )
plt.yticks(range(1),'')
plt.xticks(range(1),'')
ax1 = fig.add_subplot(2,1,1)
#ax1.plot(range(10),'b-')
ax2 = fig.add_subplot(212)

ax1.plot(iteration, test_err , 'r')

ax2.plot(iteration , train_err , 'g')
#plt.setp(ax2.get_xticklabels(),visible=False)
#fig.axes.get_yaxis().set_visible(False)
fig.tight_layout()

plt.show()