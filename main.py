# Import packages
import os, csv, cv2
import numpy as np
from scipy import ndimage

# Loading data
lines = []
with open("../data/data_22_06/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
images = []
labels = []
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = '../data/data_22_06/IMG/' + filename
    image = ndimage.imread(current_path)  
    images.append(image)
    
    labels.append(float(line[3]))
 
print("Images length is", len(images))
print("Labels length is", len(labels), "minimum value is {0} and maximum valus is {1}".format(min(labels), max(labels)))


## Data Augmentation
augmented_images, augmented_labels = [], []
for image, label in zip(images, labels):
    augmented_images.append(image)
    augmented_labels.append(label)
    augmented_images.append(cv2.flip(image,1))
    augmented_labels.append(label*-1.0)

X_train = np.array(images)
Y_train = np.array(labels)

###--------------------------------- Neural Network Model ------------------------------------------------------ ###
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Lambda, Input
from keras.preprocessing.image import ImageDataGenerator

debug = False
batch_size = 32
epochs = 2

# PreProcessing input data
train_datagen = ImageDataGenerator(rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    preprocessing_function=preprocess_input) 

train_datagen.fit(X_train)

train_generator = train_datagen.flow(
    X_train, 
    Y_train, 
    batch_size=batch_size)

'''
validation_generator = train_datagen.flow(
    x_train, 
    y_train, 
    batch_size=batch_size,
    subset='validation') # set as validation data
'''

# Create the base pre-trained model
inception = InceptionV3(weights='imagenet', include_top=False,
                        input_shape=(160,320,3))

# Freeze all convolutional InceptionV3 layers
for layer in inception.layers:
    layer.trainable = False

# Input Layer
inputs = Input(shape=(160,320,3))

# Pre-process the input with Kera's Lambda layer
preprocessed_input = Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3))

# Add the top layers
x = preprocessed_input(inputs)
x = inception(x)
## test x = inception.output
end_pooling = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(end_pooling)
prediction = Dense(1, activation='linear')(x)

# Build the model to train
## test model = Model(inputs=inception.input, outputs=prediction)
model = Model(inputs=inputs, outputs=prediction)

# compile the model 
model.compile(optimizer='rmsprop', loss='mean_squared_error')

if debug:
    print("Model summary:")
    # Check the summary of this new model to confirm the architecture
    model.summary()

# Train the model
model.fit(
    X_train,
    Y_train,
    batch_size = batch_size ,
    validation_split = 0.2,
    epochs = epochs, 
    shuffle = True)

# Save the model
model.save('model.h5')

