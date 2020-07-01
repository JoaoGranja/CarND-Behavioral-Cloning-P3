# Import packages
import os, csv, cv2
import numpy as np
from scipy import ndimage
import tensorflow as tf
import sklearn
from math import ceil
from random import shuffle
import matplotlib.pyplot as plt

### ---------------------------------------------- Data Generator ------------------------------------------ ###
def generator(samples, batch_size=32):
    correction = [0, 0.2, -0.2]
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            labels = []
            for batch_sample in batch_samples:
                for i in range(3):
                    filename = batch_sample[i].split('/')[-1]
                    
                    if len(batch_sample[i].split('/')) > 2: # The training data from Udacity has a different format
                        data_dir = batch_sample[i].split('/')[3]
                    else:
                        data_dir = "data"
                                            
                    current_path = "/opt/carnd_p3/" + data_dir +'/IMG/' + filename
                
                    image = ndimage.imread(current_path)  
                    yuv=cv2.cvtColor(image,cv2.COLOR_RGB2YUV)
                    
                    images.append(yuv)
                    labels.append(float(line[3]) + correction[i])

            ## Data Augmentation
            augmented_images, augmented_labels = [], []
            for image, label in zip(images, labels):
                augmented_images.append(image)
                augmented_labels.append(label)
                augmented_images.append(cv2.flip(image,1))
                augmented_labels.append(label*-1.0)


            X_train = np.array(augmented_images)
            y_train = np.array(augmented_labels)
            yield sklearn.utils.shuffle(X_train, y_train)

### ---------------------------------------------- Loading Data ------------------------------------------ ###
# Loading data from several sources
source_paths = ["/opt/carnd_p3/data_29_06/"]

samples = []
for path in source_paths:
    with open(path+"driving_log.csv") as csvfile:
        reader = csv.reader(csvfile)
        if path == "/opt/carnd_p3/data/":
            next(reader)
        for line in reader:
            samples.append(line)

### ---------------------------------------------- Traning and Validation Data Split ------------------------------------------ ###     
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

print("Train samples length is", len(train_samples))
print("Validation samples length is", len(validation_samples))

###--------------------------------- Neural Network Model ------------------------------------------------------ ###
from keras.models import Model, Sequential
from keras.layers import Dense, Lambda, Flatten, Conv2D, MaxPooling2D, Activation, Cropping2D
from keras.layers import Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard

debug = True
batch_size = 32
epochs = 5

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)
    
    
# Build a Sequential Model
model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x - 128) / 128))
# Conv 1
model.add(Conv2D(filters=24, kernel_size=(5,5), strides=(2,2), padding='valid'))
model.add(Activation('relu'))

# Conv 2
model.add(Conv2D(filters=36, kernel_size=(5,5), strides=(2,2), padding='valid'))
model.add(Activation('relu'))

# Conv 3
model.add(Conv2D(filters=48, kernel_size=(5,5), strides=(2,2), padding='valid'))
model.add(Activation('relu'))

# Conv 4
model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))

# Conv 5
model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))

model.add(Flatten())

# Fully Connected 1
model.add(Dense(1000))
# Fully Connected 2
model.add(Dense(100))
# Fully Connected 3
model.add(Dense(1))

# compile the model 
model.compile(optimizer='adam', loss='mse')

if debug:
    print("Model summary:")
    # Check the summary of this new model to confirm the architecture
    model.summary()

### --------------------------------- Train and save the model ------------------------------------------------------ ###
tensorboard_callback = TensorBoard(log_dir="./logs")

history_object = model.fit_generator(train_generator, 
                    steps_per_epoch=ceil(len(train_samples)/batch_size), 
                    validation_data=validation_generator, 
                    validation_steps=ceil(len(validation_samples)/batch_size), 
                    epochs=epochs, verbose=1,
                    callbacks=[tensorboard_callback])

# Save the model
model.save('model.h5')


### ---------------------------------------------- Plot Training and Validation Results ----------------------- ###
if debug:
    # print the keys contained in the history object
    print(history_object.history.keys())

    # plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    
    plt.savefig(os.path.join("examples", "model_loss"))
    plt.show()
    plt.close()
