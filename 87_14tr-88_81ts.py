#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 19:20:59 2019

@author: juangabriel
"""

# Redes Neuronales Convolucionales

# Instalar Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Instalar Tensorflow y Keras
# conda install -c conda-forge keras


# Parte 1 - Construir el modelo de CNN

# Importar las liobrerías y paquetes
from keras.models import Sequential
from keras.layers import  Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# from keras.models import Sequential
# from keras.layers import Dense
from keras.callbacks import EarlyStopping
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score
from matplotlib import pyplot as plt
#Îfrom keras.utils import np_utils
import numpy as np
import random
import tensorflow as ts  

random.seed(42)
np.random.seed(42)
ts.random.set_seed(42)

# Inicializar la CNN
classifier = Sequential()

# Paso 1 - Convolución
classifier.add(Conv2D(filters = 32,kernel_size = (3, 3), 
                      input_shape = (64, 64, 3), activation = "relu"))

# Paso 2 - Max Pooling
classifier.add(MaxPooling2D(pool_size = (2,2)))

# Una segunda capa de convolución y max pooling
classifier.add(Conv2D(filters = 32,kernel_size = (3, 3), activation = "relu"))

classifier.add(MaxPooling2D(pool_size = (2,2)))


#este dropout desactiva el 25% de las conexiones entre las neuronas, lo cual mejora los resultados
classifier.add(Dropout(0.25 )) # NUEVO 

# Paso 3 - Flattening
#classifier.add(Flatten())

#################################################################

classifier.add(Conv2D(filters = 32,kernel_size = (3, 3), 
                      input_shape = (64, 64, 3), activation = "relu"))

# Paso 2 - Max Pooling
classifier.add(MaxPooling2D(pool_size = (2,2)))

# Una segunda capa de convolución y max pooling
classifier.add(Conv2D(filters = 32,kernel_size = (3, 3), activation = "relu"))

classifier.add(MaxPooling2D(pool_size = (2,2)))


#este dropout desactiva el 25% de las conexiones entre las neuronas, lo cual mejora los resultados
classifier.add(Dropout(0.25 )) # NUEVO 

# Paso 3 - Flattening
classifier.add(Flatten())
################################################################################

# Paso 4 - Full Connection

classifier.add(Dense(units = 128, activation = "relu"))
#classifier.add(Dense(units = 64, activation = "relu")) #NUEVO
classifier.add(Dense(units = 32, activation = "relu")) #NUEVO
#este dropout desactiva el 25% de las conexiones entre las neuronas, lo cual mejora los resultados
classifier.add(Dropout(0.25)) # NUEVO 
classifier.add(Dense(units = 1, activation = "sigmoid"))

# Compilar la CNN
classifier.compile(optimizer = "RMSprop", loss = "binary_crossentropy", metrics = ["accuracy"]) #NUEVO
#classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
# Parte 2 - Ajustar la CNN a las imágenes para entrenar 
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_dataset = train_datagen.flow_from_directory('dataset/training_set',
                                                    target_size=(64, 64),
                                                    batch_size=32,
                                                    class_mode='binary')

testing_dataset = test_datagen.flow_from_directory('dataset/test_set',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')

#NUEVO #NUEVO #NUEVO #NUEVO #NUEVO #NUEVO #NUEVO #NUEVO #NUEVO #NUEVO
# A los 15 valores de coste sin variar significativamente deja de ajustar el modelo
earlystopper = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=100, verbose=1, mode='auto')

# Ajusta el modelo a más de 1000 iteraciones con el 'earlystopper' y lo asigna al historial
history = classifier.fit_generator(training_dataset,
                        steps_per_epoch=350,
                        epochs=150,
                        validation_data=testing_dataset,
                        validation_steps=200,
                        callbacks = [earlystopper])

# Plots 'history'
history_dict=history.history
loss_values = history_dict['loss']
val_loss_values=history_dict['val_loss']
plt.plot(loss_values,'bo',label='training loss') 
plt.plot(val_loss_values,'r',label='training loss val')
plt.show()


# Parte 3 - Cómo hacer nuevas predicciones
#import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_dataset.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
    





