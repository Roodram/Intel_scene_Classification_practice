# Import statements
from skimage.io import imread
import matplotlib.pyplot as plt
import numpy as np
import os
from skimage.transform import resize
from keras.utils import to_categorical
from sklearn.utils import shuffle
import keras
from keras.models import Model
from keras.layers import Dense, Flatten, Input, Activation, Conv2D, MaxPooling2D, BatchNormalization, Dropout
import numpy as np

# Data preparation for train and test

def prepare_data(dataType, dataLength):
    X = np.zeros((dataLength, 150, 150, 3), dtype=np.uint8)
    Y = np.zeros((dataLength, 1), dtype=np.uint8)
    image_dir = "Dataset/seg_" + dataType + "/"

    i=0
    j=-1

    for _,folders,_ in os.walk(image_dir):
        if(j>0):
            break
        for folder in folders:
            j+=1
            folder = image_dir + folder + "/"
            for _,_,images in os.walk(folder):
                for image in images:
                    img = imread(folder+image)
                    Y[i] = j
                    try:
                        X[i] = img
                    except ValueError:
                        shape = img.shape
                        X[i, :shape[0], :shape[1], :] = img
                    i+=1
                    
    print("Shape of X : " + str(X.shape))
    print("Shape of Y : " + str(Y.shape))
    Y_oh = to_categorical(Y)
    print("Shape of One hot encoded Y : " + str(Y_oh.shape))
    
    X, Y_oh = shuffle(X, Y_oh)
    print("\nShuffled X and Y")
    print("\nExample image :")
    plt.imshow(X[4,:,:,:])
    plt.show()
    print(Y([4]))
    
    np.save("Models/X_" + dataType, X)
    np.save("Models/Y_" + dataType, Y_oh)
    
    print("\nData saved")
    

train_data_len = 14034
prepare_data("train", train_data_len)

test_data_len = 3000
prepare_data("test", test_data_len)

# Traning of train data

X_train = np.load("Models/X_train.npy")
Y_train_oh = np.load("Models/Y_train.npy")


def CNN_Model():
    inputs = Input((150, 150, 3, ))
    
    X = Conv2D(8, (5, 5), padding='valid')(inputs)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(2)(X)
    
    X = Conv2D(16, (5, 5), padding='valid')(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(2)(X)
    X = Dropout(0.25)(X)
    
    X = Conv2D(32, (3, 3), padding='valid')(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(2)(X)
        
    X = Conv2D(64, (3, 3), padding='valid')(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(2)(X)
    X = Dropout(0.25)(X)
    
    X = Conv2D(128, (3, 3), padding='valid')(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(2)(X)
    
    X = Conv2D(256, (1, 1), padding='valid')(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(2)(X)
    X = Dropout(0.25)(X)
    
    X = Flatten()(X)
    X = Dense(64, activation='tanh')(X)
    X = Dense(6 , activation='softmax')(X)
    
    model = Model(inputs=inputs, outputs=X)
    return model


model = CNN_Model()
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, Y_train_oh, batch_size=20, epochs=2, verbose=1,shuffle=True, callbacks=[keras.callbacks.History()])

# Testing on validation data

X_test = np.load("Models/X_test.npy")
Y_test_oh = np.load("Models/Y_test.npy")

history = model.evaluate(X_test, Y_test_oh)

print(history)

# Making predictions on test data

pred_data_len = 7300
image_dir = "Dataset/seg_pred/"

i=0
image_file = open("Data/PredResult.csv", "w")
f = csv.writer(image_file)
f.writerow(["image_name","label"])
X_prediction = np.zeros((1, 150, 150, 3))
for _,_,images in os.walk(image_dir):
    for image in images:
        img = imread(image_dir+image)
        try:
            shape = img.shape
            X_prediction[0, :shape[0], :shape[1], :] = img
        except ValueError:
            print("Image size out of bounds")
            print("Error image = " + image)
            plt.imshow(img)
            plt.show()
            continue
        f.writerow([image,np.argmax(model.predict(X_prediction))])
        i+=1
