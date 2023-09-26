import random
import numpy as np
import matplotlib.pyplot as plt

class PerceptronProfe:
    grad_pesos=[]
    def __init__(self, n):
        self.n = n
        self.bias =np.random.rand()
        self.peso1 =np.random.rand()
        self.peso2 =np.random.rand()
        self.peso3 =np.random.rand()
        self.tasaAprendizaje=0.5

        self.historialPeso1=[self.peso1]
        self.historialPeso2=[self.peso2]
        self.historialPeso3=[self.peso3]
        
        #ESCALON
    def propagacion(self,entradaPeso1,entradaPeso2,entradaPeso3):
        
        respuesta = self.peso1 * entradaPeso1 + self.peso2 * entradaPeso2 + self.peso3 * entradaPeso3 + self.bias

        if respuesta>=0:
            respuesta=1
            
        elif(respuesta<0):
            respuesta=0      
        #print(respuesta)  
        
        return respuesta
    def limpiarHistorial(self):
        self.historialPeso1=[self.peso1]
        self.historialPeso2=[self.peso2]
        self.historialPeso3=[self.peso3]
    
    def entrenador(self, entradaPesoR,entradaPesoG,entradaPesoB,respuestaDeseada):
        #print(f"PRIMER PESO R {self.peso1} PRIMER PESO G {self.peso2} PRIMER PESO B  {self.peso3}   PRIMER BIAS  {self.bias}")
        errores =0
        numeroDeEpocas=100
        for epoca in range(numeroDeEpocas):
            
            for i in range(len(entradaPesoR)):
                
                respuestaActual = self.propagacion(entradaPesoR[i],entradaPesoG[i],entradaPesoB[i])
                
                error = respuestaDeseada[i]- respuestaActual
                #Actualizacion de bias                    
                self.bias -= self.tasaAprendizaje*error
                #Actualizacion de pesos r g b
                self.peso1 = self.peso1 + (self.tasaAprendizaje * error * entradaPesoR[i])#R
                self.peso2 = self.peso2 + (self.tasaAprendizaje * error * entradaPesoG[i])#G
                self.peso3 = self.peso3 + (self.tasaAprendizaje * error * entradaPesoB[i])#B      
                   
                self.historialPeso1.append(self.peso1)
                self.historialPeso2.append(self.peso2)
                self.historialPeso3.append(self.peso3)  
                
                #print(f" Actualizando peso  R {self.peso1}     actualizando peso G  {self.peso2}   actualizando peso B {self.peso3}   actualizando bias   {self.bias}")
                if not np.array_equal(respuestaActual,respuestaDeseada[i]):
                    #print(f"Respuesta del pc   {respuestaActual} es diferente de  respuestaActual {respuestaDeseada[i]}")
                    errores+=1
        #print("     FALLOS    " ,errores)    
        #print(f" PESO  FINAL  R {self.peso1}     PESO FINAL G  {self.peso2}   PESO FINAL B {self.peso3}   FINAL BIAS   {self.bias}")
                
    
    def mostrarGraficoPesos(self):
        plt.plot(self.historialPeso1, label="Peso 1")
        plt.plot(self.historialPeso2, label="Peso 2")
        plt.plot(self.historialPeso3, label="Peso 3")
        plt.xlabel("Epoca")
        plt.ylabel("Valor del peso")
        plt.title("Evolución de los pesos durante el Entrenamiento")
        plt.legend()
        plt.show()
            