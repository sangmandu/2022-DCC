import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
import keras
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import classification_report
from keras.models import load_model
categories = ["L2_3", "L2_10", "L2_12", "L2_15", "L2_20", "L2_21", "L2_24", "L2_25", "L2_27", "L2_30","L2_33", "L2_34", "L2_39", "L2_40", "L2_41", "L2_44", "L2_45","L2_46", "L2_50", "L2_52"]

def make_npy(image_w,image_h , groups_folder_path = './data'):
    X = []
    Y = []
    for idex, categorie in tqdm(enumerate(categories)):
        # label = [0 for i in range(num_classes)]
        label = idex
        image_dir = groups_folder_path  + '/' + categorie + "/"
    
        for top, dir, f in os.walk(image_dir):

            for filename in f:
                img = cv2.imread(image_dir+filename)
                img = cv2.resize(img,  (image_w, image_h))
                X.append(img/256)
                Y.append(label)
    
    X = np.array(X)
    Y = np.array(Y)
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y)
    xy = (X_train, X_test, Y_train, Y_test)
    
    np.save("./img_data.npy", xy)
    return

def make_model(image_w,image_h,num_classes=20):
    input_shape = (image_w, image_h, 3)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (2, 2), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()

    return model
def fit_model(model,x_train, x_test, y_train, y_test,batch_size,epochs,num_classes,model_name="BasicCNN"):
    ## y to_categorical
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=2,)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    
    model.save(f'{model_name}.h5')
    return model
def load_BasicCNN():
    return load_model('BasicCNN.h5')

def get_f1_score(model,x_test,y_test):
    y_pred = model.predict(x_test, verbose=2) 
    y_pred=np.argmax(y_pred, axis=1)
    print(classification_report(y_test, y_pred))
    return
