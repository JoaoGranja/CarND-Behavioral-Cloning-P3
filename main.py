# Import packages
import os, csv, cv2
import numpy as np
from scipy import ndimage
import tensorflow as tf

# Loading data from several sources
source_paths = ["../data/data_22_06/", "../data/data_26_06/","data/"]
correction = [0, 0.2, -0.2]
images = []
labels = []
for path in source_paths:
    print(path)
    lines = []
    with open(path+"driving_log.csv") as csvfile:
        reader = csv.reader(csvfile)
        if path == "data/":
            next(reader)
        for line in reader:
            lines.append(line)

    for line in lines:
        for i in range(3):
            source_path = line[i]
            filename = source_path.split('/')[-1]
            current_path = path +'IMG/' + filename
            image = ndimage.imread(current_path)  
            images.append(image)

            labels.append(float(line[3]) + correction[i])

## Data Augmentation
augmented_images, augmented_labels = [], []
for image, label in zip(images, labels):
    augmented_images.append(image)
    augmented_labels.append(label)
    augmented_images.append(cv2.flip(image,1))
    augmented_labels.append(label*-1.0)

X_train = np.array(augmented_images)
Y_train = np.array(augmented_labels)

print("Images length is", len(augmented_images))
print("Labels length is", len(augmented_labels), "minimum value is {0} and maximum valus is {1}".format(min(augmented_labels), max(augmented_labels)))


###--------------------------------- Neural Network Model ------------------------------------------------------ ###
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image
from keras.models import Model, Sequential
from keras.layers import Dense, GlobalAveragePooling2D, Lambda, Input, Flatten, Conv2D, MaxPooling2D, Activation, Cropping2D
from keras.layers import Dropout
from keras.preprocessing.image import ImageDataGenerator

debug = True
batch_size = 32
epochs = 2

'''
# Input Layer
inputs = Input(shape=(160,320,3))

# Create the base pre-trained model
inception_model = InceptionV3(weights='imagenet', include_top=False,
                        input_shape=(160,320,3))

# Freeze all convolutional InceptionV3 layers
for layer in inception_model.layers:
    layer.trainable = False

# Pre-process the input with Kera's Lambda layer
preprocessed_inputs = Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3))


tf.reset_default_graph()

# Add the top layers
#x = preprocessed_inputs(inputs)
x = inception_model(inputs)
#x = inception_model.output
end_pooling = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(end_pooling)
prediction = Dense(1, activation='linear')(x)

# Build the model to train
## test model = Model(inputs=inception.input, outputs=prediction)
model = Model(inputs=inputs, outputs=prediction)
'''
 
# Build a Sequential Model
model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))
model.add(Conv2D(filters=32, kernel_size=(5,5), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Max Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
model.add(Dropout(0.5))

model.add(Conv2D(filters=96, kernel_size=(5,5), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Max Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
model.add(Dropout(0.5))

model.add(Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Max Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(512))
model.add(Dense(84))
model.add(Dense(1))

# compile the model 
#model.compile(optimizer='rmsprop', loss='mean_squared_error')
model.compile(optimizer='adam', loss='mse')

if debug:
    print("Model summary:")
    # Check the summary of this new model to confirm the architecture
    model.summary()

# Train the model
model.fit(
    X_train,
    Y_train,
    validation_split = 0.2,
    epochs = epochs, 
    shuffle = True)

# Save the model
model.save('model.h5')

'''
### print the keys contained in the history object
print(model.history.keys())

### plot the training and validation loss for each epoch
plt.plot(model.history['loss'])
plt.plot(model.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
'''