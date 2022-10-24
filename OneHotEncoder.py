from keras.datasets import mnist
import numpy as np
from sklearn.svm import SVC
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pickle
import glob
import cv2

def ohefunc(data, n):
    out = []
    for i in range(len(data)):
        curr = []
        for j in range(n):
            if j==data[i]:
                curr.append(1)
            else:
                curr.append(0)
        out.append(curr)
    return out

(X_train,Y_train),(X_test,Y_test) = mnist.load_data()
X_train = X_train.tolist()
Y_train = Y_train.tolist()
X_test = X_test.tolist()
Y_test = Y_test.tolist()

X_train = X_train + X_test
Y_train = ohefunc(Y_train + Y_test, 25)


f = open("./Data/Subsets/DAT_Representation/dataset_augmented.dat", "rb+")
dataset = pickle.load(f)

# english = {'0':0, '1':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9}
Devnagari = {"0":0,"1":10,"2":2,"3":3,"4":11,"5":12,"6":13,"7":14,"8":15,"9":16}
Western_Arabic = {"0":17,"1":1,"2":18,"3":19,"4":20,"5":21,"6":22,"7":23,"8":24,"9":9}
dw = [[], []]
for i in range(len(dataset[0])):
    image = (255-dataset[0][i]).tolist()
    data = dataset[1][i]
    number = str(data[1])
    dw[0].append(image)
    if data[0] == "D":
        # print(Devnagari[number])
        dw[1].append(Devnagari[number])
    else:
        # print(Western_Arabic[number])
        dw[1].append(Western_Arabic[number])

# Y = pd.DataFrame(y_test)
# Y[0].unique()
#
# Y = pd.DataFrame(Y_train)
# ohe = OneHotEncoder(sparse=False)
# y_test_ = ohe.fit_transform(Y)
#
# print(np.shape(y_test_))

dw[1] = ohefunc(dw[1], 25)

X_train = X_train + dw[0]
Y_train = Y_train + dw[1]

# xtrain = np.array(X_train)
# ytrain = np.array(Y_train)

# print(np.shape(xtrain), np.shape(ytrain))

dataset_final = [X_train, Y_train]

file2 = open("./Data/Subsets/DAT_Representation/dataset_final.dat", "wb+")
pickle.dump(dataset_final, file2)
file2.close()


for i in range(len(X_train)):
    j = i*100
    cv2.imshow('asdf', cv2.resize(np.array(X_train[j]).astype(np.uint8), (300, 300)))
    cv2.waitKey(10)
