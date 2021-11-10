
import sys   
import os 
import tensorflow as tf
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.python.keras import optimizers 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras import backend as k

k.clear_session()

#Almacenamiento de imagenes
data_entrenamiento = r'C:/Users/jessd/OneDrive/Escritorio/data/entrenamiento'
data_validacion = r'C:/Users/jessd/OneDrive/Escritorio/data/validacion'

#Parametros 
epocas=30
altura, longitud = 100,100
batch_size = 5  
pasos = 10   
pasos_validacion = 200  
FiltrosConv1= 32    
FiltrosConv2= 64     
Tamaño_filtro1 = (3,3)  
Tamaño_filtro2 = (2,2)  
tamaño_pool=(2,2)   
clases = 2
Lr = 0.0005 

#Pre-procesamiento de imagenes
entrenamiento_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.3,    
    zoom_range=0.3,     
    horizontal_flip=True
    )   

Validacion_datagen=ImageDataGenerator(
    rescale=1./255
    )

imagen_entrenamiento = entrenamiento_datagen.flow_from_directory(
    data_entrenamiento,    
    target_size=(altura,longitud),  
    batch_size= batch_size,
    class_mode='categorical',   
    )

imagen_validacion= Validacion_datagen.flow_from_directory(
    data_validacion,
    target_size=(altura,longitud),
    batch_size=batch_size,
    class_mode='categorical'
    )

#Red Neuronal
cnn = Sequential()  

cnn.add(Conv2D(FiltrosConv1,Tamaño_filtro1, padding='same', input_shape=(altura,longitud,3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=tamaño_pool)) 

cnn.add(Conv2D(FiltrosConv2 ,Tamaño_filtro2, padding='same', activation='relu'))
cnn.add(MaxPooling2D(pool_size=tamaño_pool))

cnn.add(Flatten())  
cnn.add(Dense(256,activation='relu')) 
                                    
cnn.add(Dropout(0.5))                          
cnn.add(Dense(clases,activation='softmax')) 
                
#Paremetros para optimizar nuestro algoritmo
cnn.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=Lr), metrics=['accuracy'])   
#cnn.compile(loss='categorical_crossentropy', 
            #optimizer=tf.keras.optimizers.Adam(lr=Lr), 
            #metrics=['accuracy']) 

# Entrenamiento del algoritmo 
cnn.fit(imagen_entrenamiento, steps_per_epoch=pasos, epochs=epocas,validation_data = imagen_validacion, validation_steps=pasos_validacion) 
                                            
print('Modelo entrenado')                                            
                                            
dir = './modelo/' 

if not os.path.exists(dir):
    os.makedirs(dir)

cnn.save('./modelo/modelo.h5') 
cnn.save_weights('./modelo/pesos.h5') 

