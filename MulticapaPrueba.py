import numpy as np
import json
from sklearn.preprocessing import OneHotEncoder

class Multicapaprueba():

    def __init__(self, datosEntrenamiento):
        self.tasa_aprendizaje =0.01
        
        np.random.seed(0)  # Fijamos la semilla para reproducibilidad

        self.num_neuronas_capa_oculta = 3  # Número de neuronas en la capa oculta
        self.num_neuronas_salida = 5  # Número de clases de salida
         # Inicialización de pesos y bias para la capa oculta y la capa de salida
        self.pesos_ocultos = np.random.randn(self.num_neuronas_capa_oculta, self.num_neuronas_salida)
        self.bias_oculto = np.zeros((1, self.num_neuronas_salida))
        self.pesos_salida = np.random.randn(self.num_neuronas_salida, self.num_neuronas_salida)
        self.bias_salida = np.zeros((1, self.num_neuronas_salida))
        self.X_normalized =datosEntrenamiento
        self.epocas =100
        self.losses=[]
        #self.clases=datosClase
        self.clases = np.array([[1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0],
             [0, 1, 0, 0, 0], [0, 1, 0, 0, 0], [0, 1, 0, 0, 0], [0, 1, 0, 0, 0], [0, 1, 0, 0, 0],
             [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0],
             [0, 0, 0, 1, 0], [0, 0, 0, 1, 0], [0, 0, 0, 1, 0], [0, 0, 0, 1, 0], [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1], [0, 0, 0, 0, 1], [0, 0, 0, 0, 1], [0, 0, 0, 0, 1], [0, 0, 0, 0, 1]])
        
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self,x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    def softmax(self,x):
        valor_exponentes = np.exp(x - np.max(x))
        return valor_exponentes / np.sum(valor_exponentes, axis=1, keepdims=True)
    
    def entrenamiento(self):
        for epoca in range(self.epocas):
            self.activacion_capa_oculta = np.dot(self.X_normalized, self.pesos_ocultos) + self.bias_oculto
            self.salida_capa_oculta = self.sigmoid(self.activacion_capa_oculta)
            self.activacion_capa_oculta = np.dot(self.salida_capa_oculta, self.pesos_salida) + self.bias_salida
            self.prediccion_salida = self.softmax(self.activacion_capa_oculta)
                    # Cálculo de la pérdida usando la entropía cruzada
            loss = -np.sum(self.clases * np.log(self.prediccion_salida)) / len(self.data_input)
            self.losses.append(loss)
                    # Backpropagation
            gradiente_perdida = self.prediccion_salida - self.clases
            gradiente_pesos_salida = np.dot(self.salida_capa_oculta.T, gradiente_perdida)
            gradiente_bias_salida = np.sum(gradiente_perdida, axis=0, keepdims=True)
            gradiente_perdida_oculta = np.dot(gradiente_perdida, self.pesos_salida.T) * self.sigmoid_derivative(self.salida_capa_oculta)
            gradiente_pesos_ocultos = np.dot(self.X_normalized.T, gradiente_perdida_oculta)  # Fix here
            gradiente_bias_oculto = np.sum(gradiente_perdida_oculta, axis=0, keepdims=True)

        # Actualización de pesos y bias
            self.pesos_salida -= self.tasa_aprendizaje * gradiente_pesos_salida
            self.bias_salida -= self.tasa_aprendizaje * gradiente_bias_salida
            self.pesos_ocultos -= self.tasa_aprendizaje * gradiente_pesos_ocultos
            self.bias_oculto -= self.tasa_aprendizaje * gradiente_bias_oculto
            if epoca %50 ==0:
                print(f" EPOCA {epoca}, fallos : {loss}")
    
    
    def clasificacion(self,datos):
        self.activacion_capa_oculta = np.dot(datos,self.pesos_ocultos) + self.bias_oculto
        self.salida_capa_oculta = self.sigmoid(self.activacion_capa_oculta)
        self.activacion_capa_salida = np.dot(self.salida_capa_oculta, self.pesos_salida) + self.bias_salida
        self.prediccion_salida = self.softmax(self.activacion_capa_salida)
        
        # Clasificación
        self.prediccion = np.argmax(self.prediccion_salida, axis=1)[0]
        
        print(f'la prediccion es {self.prediccion}')

