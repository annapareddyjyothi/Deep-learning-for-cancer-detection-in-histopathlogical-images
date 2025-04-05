import os
import cv2
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.models import model_from_json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.callbacks import ModelCheckpoint
import os
from keras.optimizers import Adam
from keras.optimizers import SGD
# Non-Binary Image Classification using Convolution Neural Networks


path = 'BreaKHis_v1'
'''
labels = []
X = []
Y = []

def getID(name):
    index = 0
    for i in range(len(labels)):
        if labels[i] == name:
            index = i
            break
    return index        
    

for root, dirs, directory in os.walk(path):
    for j in range(len(directory)):
        name = os.path.basename(root)
        if name not in labels:
            labels.append(name)
print(labels)

for root, dirs, directory in os.walk(path):
    for j in range(len(directory)):
        name = os.path.basename(root)
        if 'Thumbs.db' not in directory[j]:
            img = cv2.imread(root+"/"+directory[j])
            img = cv2.resize(img, (64,64))
            im2arr = np.array(img)
            im2arr = im2arr.reshape(64,64,3)
            label = getID(name)
            X.append(im2arr)
            Y.append(label)
            print(name+" "+root+"/"+directory[j]+" "+str(label))
        
X = np.asarray(X)
Y = np.asarray(Y)
print(Y)

np.save('model/X.txt',X)
np.save('model/Y.txt',Y)
'''
X = np.load('model/X.txt.npy')
Y = np.load('model/Y.txt.npy')

X = X.astype('float32')
X = X/255
    
test = X[3]
cv2.imshow("aa",test)
cv2.waitKey(0)
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]
Y = to_categorical(Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1) #split dataset into train and test

print(Y)
sgd_cnn = Sequential()
sgd_cnn.add(Convolution2D(32, 3, 3, input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
sgd_cnn.add(MaxPooling2D(pool_size = (2, 2)))
sgd_cnn.add(Convolution2D(32, 3, 3, activation = 'relu'))
sgd_cnn.add(MaxPooling2D(pool_size = (2, 2)))
sgd_cnn.add(Flatten())
sgd_cnn.add(Dense(output_dim = 256, activation = 'relu'))
sgd_cnn.add(Dense(output_dim = y_train.shape[1], activation = 'softmax'))
opt = SGD(lr = 0.001)
sgd_cnn.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])
if os.path.exists("model/sgd_cnn_weights.hdf5"):
    sgd_cnn.load_weights('model/sgd_cnn_weights.hdf5')
else:
    model_check_point = ModelCheckpoint(filepath='model/sgd_cnn_weights.hdf5', verbose = 1, save_best_only = True)
    hist = sgd_cnn.fit(X_train, y_train, epochs=30, shuffle=True, verbose=2, validation_data=(X_test, y_test), callbacks=[model_check_point])
    f = open('model/sgd_history.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()
    
predict = sgd_cnn.predict(X_test)
predict = np.argmax(predict, axis=1)
y_test = np.argmax(y_test, axis=1)
a = accuracy_score(y_test,predict)*100    
print(a)
