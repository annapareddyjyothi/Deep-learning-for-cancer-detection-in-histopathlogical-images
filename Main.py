
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
import matplotlib.pyplot as plt
import numpy as np
from tkinter import simpledialog
from tkinter import filedialog
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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
import webbrowser
from keras.optimizers import Adam
from keras.optimizers import SGD

main = tkinter.Tk()
main.title("Early Detection of Cancer using AI") #designing main screen
main.geometry("1300x1200")

global filename
global X, Y
global X_train, X_test, y_train, y_test
accuracy = []
precision = []
recall = []
fscore = []
sensitivity = []
specificity = []

labels = ['ductal_carcinoma', 'lobular_carcinoma', 'mucinous_carcinoma']

def upload():
    global filename
    filename = filedialog.askdirectory(initialdir=".")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n");
    
    
def processDataset():
    text.delete('1.0', END)
    global X, Y
    global X_train, X_test, y_train, y_test
    X = np.load('model/X.txt.npy')
    Y = np.load('model/Y.txt.npy')
    X = X.astype('float32')
    X = X/255
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    Y = to_categorical(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1) #split dataset into train and test
    text.insert(END,"Total number of images found in dataset : "+str(len(X))+"\n")
    text.insert(END,"Total cancer labels found in dataset : "+str(labels)+"\n")
    text.insert(END,"80% images used for training and 20% for testing\n\n")
    text.insert(END,"Training Images Size = "+str(X_train.shape[0])+"\n")
    text.insert(END,"Testing Images Size = "+str(X_test.shape[0]))
    text.update_idletasks()
    names, count = np.unique(np.argmax(Y, axis=1), return_counts = True)
    height = count
    bars = labels
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.title("Images Count Found in Dataset")
    plt.xlabel("Cancer Type")
    plt.ylabel("Count")
    plt.show()

def getAccuracy(algorithm, predict, y_test):
    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    cm = confusion_matrix(y_test, predict) 

    se = cm[0,0]/(cm[0,0]+cm[0,1])
    sp = cm[1,1]/(cm[1,0]+cm[1,1])
    se = se * 100
    sp = sp * 100
    sensitivity.append(se)
    specificity.append(sp)
    text.insert(END,algorithm+" Accuracy    :  "+str(a)+"\n")
    text.insert(END,algorithm+" Precision   : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall      : "+str(r)+"\n")
    text.insert(END,algorithm+" FScore      : "+str(f)+"\n")
    text.insert(END,algorithm+" Sensitivity : "+str(se)+"\n")
    text.insert(END,algorithm+" Specificity : "+str(sp)+"\n\n")
    text.update_idletasks()
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    sensitivity.append(se)
    specificity.append(sp)    
    plt.figure(figsize =(8, 6)) 
    ax = sns.heatmap(cm, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(labels)])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()

def trainAdam():
    global X, Y
    global X_train, X_test, y_train, y_test
    sensitivity.clear()
    specificity.clear()
    accuracy.clear()
    precision.clear()
    recall.clear()
    fscore.clear()
    text.delete('1.0', END)
    adam_cnn = Sequential()
    adam_cnn.add(Convolution2D(32, 3, 3, input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
    adam_cnn.add(MaxPooling2D(pool_size = (2, 2)))
    adam_cnn.add(Convolution2D(32, 3, 3, activation = 'relu'))
    adam_cnn.add(MaxPooling2D(pool_size = (2, 2)))
    adam_cnn.add(Flatten())
    adam_cnn.add(Dense(output_dim = 256, activation = 'relu'))
    adam_cnn.add(Dense(output_dim = y_train.shape[1], activation = 'softmax'))
    opt = Adam(learning_rate=0.001)
    adam_cnn.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])
    adam_cnn.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    if os.path.exists("model/adam_cnn_weights.hdf5"):
        adam_cnn.load_weights('model/adam_cnn_weights.hdf5')
    else:
        model_check_point = ModelCheckpoint(filepath='model/adam_cnn_weights.hdf5', verbose = 1, save_best_only = True)
        hist = adam_cnn.fit(X_train, y_train, epochs=30, shuffle=True, verbose=2, validation_data=(X_test, y_test), callbacks=[model_check_point])
        f = open('model/adam_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
    
    predict = adam_cnn.predict(X_test)
    predict = np.argmax(predict, axis=1)
    ytest = np.argmax(y_test, axis=1)
    getAccuracy("AI with Adaptive Moment Estimation", predict, ytest)

def trainSGD():
    global X, Y
    global X_train, X_test, y_train, y_test
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
    ytest = np.argmax(y_test, axis=1)
    getAccuracy("AI with SGD", predict, ytest)
    
def trainBatch():
    global X, Y
    global X_train, X_test, y_train, y_test
    mini_cnn = Sequential()
    mini_cnn.add(Convolution2D(32, 3, 3, input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
    mini_cnn.add(MaxPooling2D(pool_size = (2, 2)))
    mini_cnn.add(Convolution2D(32, 3, 3, activation = 'relu'))
    mini_cnn.add(MaxPooling2D(pool_size = (2, 2)))
    mini_cnn.add(Flatten())
    mini_cnn.add(Dense(output_dim = 256, activation = 'relu'))
    mini_cnn.add(Dense(output_dim = y_train.shape[1], activation = 'softmax'))
    opt = SGD(lr = 0.001)
    mini_cnn.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])
    if os.path.exists("model/mini_cnn_weights.hdf5"):
        mini_cnn.load_weights('model/mini_cnn_weights.hdf5')
    else:
        model_check_point = ModelCheckpoint(filepath='model/mini_cnn_weights.hdf5', verbose = 1, save_best_only = True)
        hist = mini_cnn.fit(X_train, y_train, batch_size=32, epochs=30, shuffle=True, verbose=2, validation_data=(X_test, y_test), callbacks=[model_check_point])
        f = open('model/sgd_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
    
    predict = mini_cnn.predict(X_test)
    predict = np.argmax(predict, axis=1)
    ytest = np.argmax(y_test, axis=1)
    getAccuracy("AI with Mini Batch Gradient Descent", predict, ytest)

def table():
    output = '<table border=1 align=center>'
    output+= '<tr><th>Algorithm Name</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>FSCORE</th><th>Sensitivity</th><th>Specificity</th></tr>'
    output+='<tr><td>AI with ADAM</td><td>'+str(accuracy[0])+'</td><td>'+str(precision[0])+'</td><td>'+str(recall[0])+'</td><td>'+str(fscore[0])+'</td><td>'+str(sensitivity[0])+'</td><td>'+str(specificity[0])+'</td></tr>'
    output+='<tr><td>AI with SGD</td><td>'+str(accuracy[1])+'</td><td>'+str(precision[1])+'</td><td>'+str(recall[1])+'</td><td>'+str(fscore[1])+'</td><td>'+str(sensitivity[1])+'</td><td>'+str(specificity[1])+'</td></tr>'
    output+='<tr><td>AI with MiniBatch</td><td>'+str(accuracy[2])+'</td><td>'+str(precision[2])+'</td><td>'+str(recall[2])+'</td><td>'+str(fscore[2])+'</td><td>'+str(sensitivity[2])+'</td><td>'+str(specificity[2])+'</td></tr>'
    output+='</table></body></html>'
    f = open("output.html", "w")
    f.write(output)
    f.close()
    webbrowser.open("output.html",new=1)

    height = [accuracy[0], accuracy[1], accuracy[2]]
    bars = ['AI with ADAM', 'AI with SGD', 'AI with MiniBatch']
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.title("Accuracy Comparison Graph")
    plt.xlabel("Algorithm Name")
    plt.ylabel("Accuracy")
    plt.show()
    
    
font = ('times', 13, 'bold')
title = Label(main, text='Early Detection of Cancer using AI')
title.config(bg='NavajoWhite2', fg='dodger blue')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=25,width=100)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=400,y=100)
text.config(font=font1)


font1 = ('times', 12, 'bold')
uploadButton = Button(main, text="Upload Histopathological Images Dataset", command=upload)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

processButton = Button(main, text="Preprocess Dataset", command=processDataset)
processButton.place(x=50,y=150)
processButton.config(font=font1) 

adamButton = Button(main, text="Train AI with ADAM", command=trainAdam)
adamButton.place(x=50,y=200)
adamButton.config(font=font1) 

sgdButton = Button(main, text="Train AI with SGD", command=trainSGD)
sgdButton.place(x=50,y=250)
sgdButton.config(font=font1)

batchButton = Button(main, text="Train AI with MiniBatch", command=trainBatch)
batchButton.place(x=50,y=300)
batchButton.config(font=font1)

graphButton = Button(main, text="Comparison Table", command=table)
graphButton.place(x=50,y=350)
graphButton.config(font=font1) 

main.config(bg='azure2')
main.mainloop()
