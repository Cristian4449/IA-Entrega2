import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#Crear vector de entradas arreglo_entrenamiento

arreglo_entrenamiento=np.array([
    [255, 51, 51],
    [255, 92, 92],
    [255, 133, 133],
    [255, 173, 173],
    [255, 214, 214],
    [239, 243, 234], 
    [232, 237, 225], 
    [225, 231, 216], 
    [218, 225, 207], 
    [211, 219, 198],
    [125, 200, 55],
    [130, 210, 50],
    [120, 195, 60],
    [127, 205, 57],
    [122, 190, 63],
    [82, 54, 29],   
    [91, 60, 33],   
    [100, 66, 36],  
    [109, 72, 40],  
    [118, 78, 43],
    [67, 50, 33],
    [74, 56, 37],   
    [81, 61, 41],   
    [88, 66, 44], 
    [95, 71, 48],
    [80, 50, 10],
    [90, 40, 5],
    [70, 55, 15],
    [85, 45, 12],
    [75, 52, 18]])/255
arreglo_clases = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

#arreglo_entrenamiento=np.array([[1,1,0],[1,1,1],[0,0,1],[0,0,0],[0,1,0],[1,0,0]])
#arreglo_clases=[1,1,1,0,0,0]

tasa_aprendizaje = 0.2
precision = 0.00000001
epocas = 5000 #
epocas_usadas = 0
arreglo_epocas_usadas = []
arreglo_pesos_1_1 = []
arreglo_pesos_1_2 = []
arreglo_pesos_1_3 = []
arreglo_pesos_2_1 = []
arreglo_pesos_2_2 = []
arreglo_pesos_2_3 = []
arreglo_pesos_3_1 = []
arreglo_pesos_3_2 = []
arreglo_pesos_3_3 = []
arreglo_pesos_1_s = []
arreglo_pesos_2_s = []
arreglo_pesos_3_s = []
arreglo_error_red = []
# Arquitectura de la red
numero_entradas = 3 # numero de entradas
cap_ocultas = 1 # Una capa oculta
neuronas_ocultas = 3 # Neuronas en la capa oculta
neurona_salida = 1 # Neuronas en la capa de salida
# Valor de umbral o bia
umbral_neurona_salida = 1.0 # umbral en neurona de salida
umbral_neuronas_ocultas = np.ones((neuronas_ocultas,1),float) # umbral en las neuronas ocultas
# Matriz de pesos sinapticos
random.seed(0) # 
w_1 = random.rand(neuronas_ocultas,numero_entradas) #se generan los pesos para las conexiones entre las neuronas ocultas salida_obtenida las de entrada
w_2 = random.rand(neurona_salida,neuronas_ocultas) #se generan los pesos para las conexiones entre las neuronas ocultas salida_obtenida las de salida

class PerceptronMulticapa():
    # constructor
    def __init__(self,arreglo_entrenamiento,arreglo_clases,w_1,w_2,umbral_neurona_salida,umbral_neuronas_ocultas,precision,epocas,tasa_aprendizaje,neuronas_ocultas,numero_entradas,neurona_salida):
        # Variables de inicialización 
        self.arreglo_entrenamiento = np.transpose(arreglo_entrenamiento)
        self.arreglo_clases = arreglo_clases
        self.w1 = w_1
        self.w2 = w_2
        self.umbral_neurona_salida = umbral_neurona_salida
        self.umbral_neuronas_ocultas = umbral_neuronas_ocultas
        self.precision = precision
        self.epocas = epocas
        self.tasa_aprendizaje = tasa_aprendizaje
        self.numero_entradas = numero_entradas
        self.neuronas_ocultas = neuronas_ocultas
        self.neurona_salida = neurona_salida
        # Variables de aprendizaje
        self.salida_esperada = 0 # Salida deseada en iteracion actual
        self.error_red = 1 # Error total de la red en una conjunto de iteraciones
        self.Error_cuadratico = 0 # Error cuadratico medio
        self.Error_prev = 0 # Error anterior
        self.Errores = []
        self.Error_actual = np.zeros((len(arreglo_clases))) # Errores acumulados en potencial_activacion_ocultas ciclo de muestras
        self.Entradas = np.zeros((1,numero_entradas))
        self.potencial_activacion_ocultas = np.zeros((neuronas_ocultas,1)) # Potencial de activacion en neuronas ocultas
        self.funcion_activacion_ocultas = np.zeros((neuronas_ocultas,1)) # Funcion de activacion de neuronas ocultas
        self.Y = 0.0 # Potencial de activacion en neurona de salida
        self.salida_obtenida = 0.0 # Funcion de activacion en neurona de salida
        self.epocas_usadas = 0
        # Variables de retropropagacion
        self.error_real = 0
        self.delta_salida = 0.0 # delta de salida
        self.delta_neuronas_ocultas = np.zeros((neuronas_ocultas,1)) # Deltas en neuronas ocultas
        
    # Funcion Sigmoide de x
    def Sigmoide(self,data):
        return 1/(1+np.exp(-data))

    # Funcion para obtener la derivada de de la funcion Sigmoide
    def DerivadaSigmoide(self,data):
        s = self.Sigmoide(data)
        return s * (1-s)

    def Probar(self,x):
        respuesta = np.zeros((len(x),1))
        z=np.transpose(x)
        for p in range(len(x)):
            self.Entradas = z[:,p]
            self.Propagar()
            respuesta[p,:] = self.salida_obtenida
        return respuesta.tolist()
    
    def Entrenar(self):
        while(np.abs(self.error_red) > self.precision):
            self.Error_prev = self.Error_cuadratico
            for i in range(len(arreglo_clases)):
                self.Entradas = self.arreglo_entrenamiento[:,i] # Senales de entrada por iteracion
                self.salida_esperada = self.arreglo_clases[i]
                self.Propagar()
                self.Backpropagation()
                self.Propagar()
                self.Error_actual[i] = (0.3333)*((self.salida_esperada - self.salida_obtenida)**2)
            # error global de la red
            self.Error()
            self.epocas_usadas +=1
            arreglo_epocas_usadas.append(self.epocas_usadas)
            arreglo_pesos_1_1.append(self.w1[0][0])
            arreglo_pesos_1_2.append(self.w1[0][1])
            arreglo_pesos_1_3.append(self.w1[0][2])
            arreglo_pesos_2_1.append(self.w1[1][0])
            arreglo_pesos_2_2.append(self.w1[1][1])
            arreglo_pesos_2_3.append(self.w1[1][2])
            arreglo_pesos_3_1.append(self.w1[2][0])
            arreglo_pesos_3_2.append(self.w1[2][1])
            arreglo_pesos_3_3.append(self.w1[2][2])
            arreglo_pesos_1_s.append(self.w2[0][0])
            arreglo_pesos_2_s.append(self.w2[0][1])
            arreglo_pesos_3_s.append(self.w2[0][2])
            arreglo_error_red.append(self.error_red)
            # Si se alcanza potencial_activacion_ocultas mayor numero de epocas
            if self.epocas_usadas > self.epocas:
                break 
                
    
    def Propagar(self):
        # Operaciones en la primer capa
        for i in range(self.neuronas_ocultas):
            self.potencial_activacion_ocultas[i,:] = np.dot(self.w1[i,:], self.Entradas) + self.umbral_neuronas_ocultas[i,:]
        
        # Calcular la activacion de la neuronas en la capa oculta
        for j in range(self.neuronas_ocultas):
            self.funcion_activacion_ocultas[j,:] = self.Sigmoide(self.potencial_activacion_ocultas[j,:])
        
        # Calcular el potencial de activacion de la neuronas de salida
        self.Y = (np.dot(self.w2,self.funcion_activacion_ocultas) + self.umbral_neurona_salida)
        # Calcular la salida de la neurona de salida
        self.salida_obtenida = self.Sigmoide(self.Y)
    
    def Backpropagation(self):
        # Calcular el error
        self.error_real = (self.salida_esperada - self.salida_obtenida)
        # Calcular delta_salida
        self.delta_salida = (self.DerivadaSigmoide(self.Y) * self.error_real)
        # Ajustar w2
        self.w2 = self.w2 + (np.transpose(self.funcion_activacion_ocultas) * self.tasa_aprendizaje * self.delta_salida)
        # Ajustar umbral umbral_neurona_salida
        self.umbral_neurona_salida = self.umbral_neurona_salida + (self.tasa_aprendizaje * self.delta_salida)
        # Calcular delta_neuronas_ocultas
        self.delta_neuronas_ocultas = self.DerivadaSigmoide(self.potencial_activacion_ocultas) * np.transpose(self.w2) * self.delta_salida
        # Ajustar los pesos w1
        for i in range(self.neuronas_ocultas):
            self.w1[i,:] = self.w1[i,:] + ((self.delta_neuronas_ocultas[i,:]) * self.Entradas * self.tasa_aprendizaje)
        
        # Ajustar el umbral en las neuronas ocultas
        for j in range(self.neuronas_ocultas):
            self.umbral_neuronas_ocultas[j,:] = self.umbral_neuronas_ocultas[i,:] + (self.tasa_aprendizaje * self.delta_neuronas_ocultas[j,:])
        
    def Error(self):
        # Error cuadratico medio
        self.Error_cuadratico = ((1/len(arreglo_clases)) * (sum(self.Error_actual)))
        self.error_red = (self.Error_cuadratico - self.Error_prev)

#Inicializar Perceptron Multicapa
Perceptro_Multi = PerceptronMulticapa(arreglo_entrenamiento,arreglo_clases,w_1,w_2,umbral_neurona_salida,umbral_neuronas_ocultas,precision,epocas,tasa_aprendizaje,neuronas_ocultas,numero_entradas,neurona_salida)
Perceptro_Multi.Entrenar()
arreglo_prueba = np.array([[255, 133, 133],[225, 231, 216],[100, 66, 36],[120, 195, 60],[81, 61, 41]])/255
respuesta = Perceptro_Multi.Probar(arreglo_prueba)
print("Resultado sin redondear:",respuesta)
print("Resultado Redondeando:",np.round(respuesta))

plt.plot(arreglo_epocas_usadas, arreglo_pesos_1_1, label='Peso E1 a NCO1', linestyle='-', color='red')
plt.plot(arreglo_epocas_usadas, arreglo_pesos_1_2, label='Peso E1 A NCO2', linestyle='-', color='blue')
plt.plot(arreglo_epocas_usadas, arreglo_pesos_1_3, label='Peso E1 a NCO3', linestyle='-', color='green')
plt.plot(arreglo_epocas_usadas, arreglo_pesos_2_1, label='Peso E2 a NCO1', linestyle='-', color='pink')
plt.plot(arreglo_epocas_usadas, arreglo_pesos_2_2, label='Peso E2 a NCO2', linestyle='-', color='gray')
plt.plot(arreglo_epocas_usadas, arreglo_pesos_2_3, label='Peso E2 a NCO3', linestyle='-', color='orange')
plt.plot(arreglo_epocas_usadas, arreglo_pesos_3_1, label='Peso E3 a NCO1', linestyle='-', color='purple')
plt.plot(arreglo_epocas_usadas, arreglo_pesos_3_2, label='Peso E3 a NCO2', linestyle='-', color='yellow')
plt.plot(arreglo_epocas_usadas, arreglo_pesos_3_3, label='Peso E3 a NCO3', linestyle='-', color='brown')
plt.plot(arreglo_epocas_usadas, arreglo_pesos_1_s, label='Peso NCO1 a Salida', linestyle='-', color='black')
plt.plot(arreglo_epocas_usadas, arreglo_pesos_2_s, label='Peso NCO2 a Salida', linestyle='-', color='magenta')
plt.plot(arreglo_epocas_usadas, arreglo_pesos_3_s, label='Peso NCO3 a Salida', linestyle='-', color='cyan')
plt.plot(arreglo_epocas_usadas, arreglo_error_red, label='Error de red', linestyle='-', color='Silver')
plt.legend()
plt.xlabel("Épocas")
plt.ylabel("Valor")
plt.title("Gráfica valores obtenidos")
plt.show()

manzana = np.array([[255, 51, 51, 1],
    [255, 92, 92, 1],
    [255, 133, 133, 1],
    [255, 173, 173, 1],
    [255, 214, 214, 1],
    [82, 54, 29, 0],   
    [91, 60, 33, 0],   
    [100, 66, 36, 0],  
    [109, 72, 40, 0],  
    [118, 78, 43, 0]])

coco = np.array([[239, 243, 234, 1], 
    [232, 237, 225, 1], 
    [225, 231, 216, 1], 
    [218, 225, 207, 1], 
    [211, 219, 198, 1],
    [67, 50, 33, 0],
    [74, 56, 37, 0],   
    [81, 61, 41, 0],   
    [88, 66, 44, 0], 
    [95, 71, 48, 0]])

pera = np.array([[125, 200, 55, 1],
    [130, 210, 50, 1],
    [120, 195, 60, 1],
    [127, 205, 57, 1],
    [122, 190, 63, 1],
    [80, 50, 10, 0],
    [90, 40, 5, 0],
    [70, 55, 15, 0],
    [85, 45, 12, 0],
    [75, 52, 18, 0]])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(len(manzana)):
    if manzana[i][3] == 1:
        ax.scatter(manzana[i][0], manzana[i][1], manzana[i][2], c='#FF0000', label='Manzana comestible' if i == 0 else None)
    else:
        ax.scatter(manzana[i][0], manzana[i][1], manzana[i][2], c='#FF8787', label='Manzana no comestible' if i == 5 else None)

for i in range(len(coco)):
    if coco[i][3] == 1:
        ax.scatter(coco[i][0], coco[i][1], coco[i][2], c='#6D5736', label='Coco comestible' if i == 0 else None)
    else:
        ax.scatter(coco[i][0], coco[i][1], coco[i][2], c='#A08760', label='Coco no comestible' if i == 5 else None)

for i in range(len(pera)):
    if pera[i][3] == 1:
        ax.scatter(pera[i][0], pera[i][1], pera[i][2], c='#39FD05', label='Pera comestible' if i == 0 else None)
    else:
        ax.scatter(pera[i][0], pera[i][1], pera[i][2], c='#9EF388', label='Pera no comestible' if i == 5 else None)

ax.legend()
ax.set_xlabel('R')
ax.set_ylabel('G')
ax.set_zlabel('B')
plt.show()






#Gráfica de dispersión
#Matriz de confusión
