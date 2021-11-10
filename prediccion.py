import numpy as np
import cv2
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import matplotlib.pyplot as plt

longitud, altura= 100,100
modelo = './modelo/modelo.h5'
pesos = './modelo/pesos.h5'
cnn=load_model(modelo) 
cnn.load_weights(pesos) 

def predict(file):
    x = load_img(file, target_size=(longitud,altura))
    x = img_to_array(x)
    x= np.expand_dims(x,axis=0)
    arreglo = cnn.predict(x)  
    resultado= arreglo[0]  
    respuesta= np.argmax(resultado) 

    if respuesta== 0:
        print()
        print('Equipo no medico------------------------------')
        
    elif respuesta == 1:
        print()
        print('Equipo medico--------------------------------------------------')


muestra = predict('.jpg')