import numpy as np
import pandas as pd

class MulticapaFinal:
    
    def __init__(self,datosEntrenamiento,datosClase):
        self.tasaAprendizaje=0.3
        self.precision =0.0000001
        self.epocas=1000
        
        self.numeroEntradas=3
        self.capasOculta=1
        self.neuronasOcultas=3
        self.neuronaSalida=5
        
        self.umbralNeuronaSalida=np.ones((self.neuronaSalida,1),dtype=float)
        
        self.umbralNeuronasOcultas=np.ones((1,self.neuronasOcultas),dtype=float)
        
        np.random.seed(0)
        self.pesoOcultas=np.random.rand(self.neuronasOcultas,self.numeroEntradas)
        #self.peso_2=np.random.rand(self.neuronaSalida,self.neuronasOcultas)
        self.pesoSalidas = np.random.rand(5, 3)


        self.arregloEntrenamiento=datosEntrenamiento
        self.arregloClase=datosClase
        
        #variables de inicializacion
        self.errorCuadratico=0
        self.errores=[]
        self.errorActual=np.zeros((len(self.arregloClase)))
        self.entradas=np.zeros((1,self.numeroEntradas))
        
        self.potencialActivacionOcultas=np.zeros((3,1))#potencial de activacion en las neuronas ocultas
        
        self.funcionActivacionOculta = np.zeros((3,1))
    
        self.potencialActivacionSalidas=np.zeros((5,1))
        
        self.Y=np.zeros((5,1))#Potencial de activacion en la neurona de la salida
        self.salidaObtenida=0.0#Funcion de Activacion en neurona de salida
        self.epocasUsadas=0
        
        #Variables de retropropagacion
        self.deltaSalida = np.zeros((5, 1))
    
        self.deltaNeuronasOcultas=np.zeros((self.neuronasOcultas,1))#Deltas en neuronas ocultas
        self.Historial=[]
    
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self,x):
        return x * (1 - x)
    
    def softmax(self,x):
        valor_exponentes = np.exp(x - np.max(x))
        return valor_exponentes / np.sum(valor_exponentes, axis=1, keepdims=True)
    def propagacion(self,entradas):
        #Capa oculta
        self.potencialActivacionOcultas =np.dot(entradas,self.pesoOcultas)+self.umbralNeuronasOcultas
        self.funcionActivacionOculta =self.sigmoid(self.potencialActivacionOcultas)
        #capaSalida
        self.potencialActivacionSalidas = np.matmul(self.funcionActivacionOculta, self.pesoSalidas.T) + self.umbralNeuronaSalida
        self.potencialSalida = self.potencialActivacionSalidas[0:1,:]
        
        self.Y =self.softmax(self.potencialSalida)
        self.capSalida =self.Y 
        return self.Y
    
    
    def retropropagacion(self,entradargb,salidaDeseada):
        
        calcularErrorSalida=salidaDeseada-self.Y
        
        #Calculando los deltas de salida
        self.deltaSalida = (calcularErrorSalida * self.sigmoid_derivative(self.Y))

        #propagar los deltas hacia atras
        error_ocultas =self.deltaSalida.dot(self.pesoSalidas)
        self.deltaNeuronasOcultas=error_ocultas* self.sigmoid_derivative(self.funcionActivacionOculta)
        
        #actualizacion de pesos
        self.deltaSalida =np.transpose(self.deltaSalida)
        self.pesoSalidas += np.dot(self.deltaSalida, self.funcionActivacionOculta) * self.tasaAprendizaje
        self.umbralNeuronaSalida +=np.sum(self.deltaSalida,axis=0)*self.tasaAprendizaje
        self.pesoOcultas += entradargb.dot(self.deltaNeuronasOcultas.T) * self.tasaAprendizaje
        self.umbralNeuronasOcultas += np.sum(self.deltaNeuronasOcultas,axis=0)*self.tasaAprendizaje
        
        #print(self.deltaSalida)
        self.Historial.append([entradargb[0],entradargb[1],entradargb[2],salidaDeseada,self.deltaNeuronasOcultas[0][0],self.deltaNeuronasOcultas[0][1],self.deltaNeuronasOcultas[0][2]
                                ,self.deltaSalida[0],self.deltaSalida[1],self.deltaSalida[2],self.deltaSalida[3],self.deltaSalida[4]])
        

    def entrenar(self):
        print("Entrando")
        for epoca in range (self.epocas):
            error_total =0
            for i,entrada in enumerate(self.arregloEntrenamiento):
                    
                #propagacion hacia adelante
                salida_calculada =self.propagacion(entrada)
                #calcular el error
                error =0.333*(self.arregloClase[i]-salida_calculada)
                error_total +=np.sum(error**2)
                #print(f'Arreglo clase{self.arregloClase[i]}  {salida_calculada}')
                #hacemos retropagacion
                self.retropropagacion(entrada,self.arregloClase[i])
            
            #calculamos el error cuadratico
            self.errorCuadratico=error_total/len(self.arregloEntrenamiento)
            #registrar error para esta epoca
            self.errores.append(self.errorCuadratico)
            # Comprobar si se cumple la condición de parada
            if self.errorCuadratico < self.precision:
                print(f"Entrenamiento completado en la época {epoca}. Error: {self.errorCuadratico}")
                break
        self.sacarExcel()
            
    def sacarExcel(self):
        #R G B, CLASE DELTA ENTRADA DELTA SALIDA
        column_labels = ["R","G","B","CLASE","DELTA ENTRADA 1","DELTA ENTRADA 2","DELTA ENTRADA 3","DELTA SALIDA 1","DELTA SALIDA 2","DELTA SALIDA 3","DELTA SALIDA 4","DELTA SALIDA 5"]
        
        
        # Crea un DataFrame con los datos y las etiquetas de columna
        df = pd.DataFrame(self.Historial, columns=column_labels)
        # Opcional: Puedes guardar el DataFrame en un archivo Excel
        df.to_excel("datos_combinados.xlsx", index=False)
        
    
    
    def clasificar(self, nueva_entrada):
        # Realiza la propagación hacia adelante para clasificar la nueva entrada
        salida_calculada = self.propagacion(nueva_entrada)
        # Puedes retornar la salida calculada que representa la clasificación
        return salida_calculada



#NO ELIMINAR 
'''
# Proporcionar los datos de entrenamiento y las clases
datos_entrenamiento = np.array([[1, 0, 1], [1, 1, 0],[0,1,1],[1,1,1],[0,0,0],[0,0,0]
                                ,[0,0,1],[1,1,1],[1,1,1],[0,1,1],[1,1,1],[0,0,0]
                                ,[0,0,0],[0,0,1],[1,1,1],[1,1,1],[0,0,0],[1,1,1]
                                ,[0,1,1],[1,1,1],[0,0,0],[0,0,1],[1,1,1],[1,1,1]
                                ,[0,1,1],[1,1,1],[0,0,0],[0,0,1],[1,1,1],[1,1,1]
                                ,[0,1,1],[1,1,1],[0,0,0],[0,0,1],[1,1,1],[1,1,1]
                                ,[0,1,1],[1,1,1],[0,0,0],[0,0,1],[1,1,1],[1,1,1]
                                ,[0,1,1],[1,1,1],[0,0,0],[0,0,1],[1,1,1],[1,1,1]  # Agrega más ejemplos de entrenamiento
                                ,[0,1,1],[1,1,1],[0,0,0],[0,0,1],[1,1,1],[1,1,1]
                                ,[0,1,1],[1,1,1],[0,0,0],[0,0,1],[1,1,1],[1,1,1]
                                ,[0,1,1],[1,1,1],[0,0,0],[0,0,1],[1,1,1],[1,1,1]
                                ,[0,1,1],[1,1,1],[0,0,0],[0,0,1],[1,1,1],[1,1,1]
                                ,[0,1,1],[1,1,1],[0,0,0],[0,0,1],[1,1,1],[1,1,1]
                                ,[0,1,1],[1,1,1],[0,0,0],[0,0,1],[1,1,1],[1,1,1]
                                ,[0,1,1],[1,1,1],[0,0,0],[0,0,1],[1,1,1],[1,1,1]
                                ,[0,1,1],[1,1,1],[0,0,0],[0,0,1],[1,1,1],[1,1,1]
                                ,[0,1,1],[1,1,1],[0,0,0],[0,0,1],[1,1,1],[1,1,1]
                                ,[0,1,1],[1,1,1],[0,0,0],[0,0,1],[1,1,1],[1,1,1]
                                ,[0,1,1],[1,1,1],[0,0,0],[0,0,1],[1,1,1],[1,1,1]
                                ,[0,1,1],[1,1,1],[0,0,0],[0,0,1],[1,1,1],[1,1,1]])
                                

#clases = np.array([0, 1, 2,3,4])  # Agrega las clases correspondientes
clases = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1],[0, 0, 0, 0, 1],[0, 0, 0, 0, 1],[0, 0, 0, 0, 1]
                   ,[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1],[0, 0, 0, 0, 1],[0, 0, 0, 0, 1],[0, 0, 0, 0, 1]
                   ,[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1],[0, 0, 0, 0, 1],[0, 0, 0, 0, 1],[0, 0, 0, 0, 1]
                   ,[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1],[0, 0, 0, 0, 1],[0, 0, 0, 0, 1],[0, 0, 0, 0, 1]
                   ,[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1],[0, 0, 0, 0, 1],[0, 0, 0, 0, 1],[0, 0, 0, 0, 1]
                   ,[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1],[0, 0, 0, 0, 1],[0, 0, 0, 0, 1],[0, 0, 0, 0, 1]
                   ,[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1],[0, 0, 0, 0, 1],[0, 0, 0, 0, 1],[0, 0, 0, 0, 1]
                   ,[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1],[0, 0, 0, 0, 1],[0, 0, 0, 0, 1],[0, 0, 0, 0, 1]
                   ,[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1],[0, 0, 0, 0, 1],[0, 0, 0, 0, 1],[0, 0, 0, 0, 1]
                   ,[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1],[0, 0, 0, 0, 1],[0, 0, 0, 0, 1],[0, 0, 0, 0, 1]
                   ,[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1],[0, 0, 0, 0, 1],[0, 0, 0, 0, 1],[0, 0, 0, 0, 1]
                   ,[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1],[0, 0, 0, 0, 1],[0, 0, 0, 0, 1],[0, 0, 0, 0, 1]
                   ,[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1],[0, 0, 0, 0, 1],[0, 0, 0, 0, 1],[0, 0, 0, 0, 1]
                   ,[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1],[0, 0, 0, 0, 1],[0,0,1,0,0],[0,1,0,0,0]
                   ,[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1],[0, 0, 0, 0, 1],[0, 0, 0, 0, 1],[0, 0, 0, 0, 1]])




matrices_rgb = [dato.reshape(3, 1) for dato in datos_entrenamiento]



red_neuronal = MulticapaFinal(datos_entrenamiento, clases)

# Llamar al método de propagación con una entrada de ejemplo
entrada_ejemplo = np.array([0, 0, 0])

red_neuronal.entrenar()
print(f" RESULTADOS EN RGB   {red_neuronal.arregloEntrenamiento.shape}    DELTA SALIDA   {red_neuronal.deltaSalida.shape}      DELTA OCULTA    {red_neuronal.deltaNeuronasOcultas.shape}")
red_neuronal.sacarExcel()
print(red_neuronal.Historial)'''