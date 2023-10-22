import numpy as np

class MulticapaFinal:
    
    def __init__(self,datosEntrenamiento,datosClase):
        self.tasaAprendizaje=0.1
        self.precision =0.0000001
        self.epocas=3000
        
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
        self.salidaEsperada=0 #Salida deseada en la iteracion actual
        self.errorRed=1#Error total de la red en un conjunto de las iteracion
        self.errorCuadratico=0
        self.error_previo=0 # error anterior
        self.errores=[]
        self.errorActual=np.zeros((len(self.arregloClase)))
        self.entradas=np.zeros((1,self.numeroEntradas))
        
        self.potencialActivacionOcultas=np.zeros((3,1))#potencial de activacion en las neuronas ocultas
        
        self.funcionActivacionOculta = np.zeros((3,1))
       
        self.potencialActivacionSalidas=np.zeros((5,1))
        
        #self.potencialActivacionSalidas=np.zeros((self.neuronaSalida,1))

        self.Y=np.zeros((5,1))#Potencial de activacion en la neurona de la salida
        self.salidaObtenida=0.0#Funcion de Activacion en neurona de salida
        self.epocasUsadas=0
        
        #Variables de retropropagacion
        self.errorReal=0
        self.deltaSalida = np.zeros((5, 1))
    

        #self.deltaSalida=0.0
        
        #self.deltaSalida=np.zeros((5,1))
        
        self.deltaNeuronasOcultas=np.zeros((self.neuronasOcultas,1))#Deltas en neuronas ocultas
    
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self,x):
        return x * (1 - x)

    def propagacion(self,entradas):
        #Capa oculta
        self.potencialActivacionOcultas =np.dot(entradas,self.pesoOcultas)+self.umbralNeuronasOcultas
        self.funcionActivacionOculta =self.sigmoid(self.potencialActivacionOcultas)
        #capaSalida
        self.potencialActivacionSalidas = np.matmul(self.funcionActivacionOculta, self.pesoSalidas.T) + self.umbralNeuronaSalida
        self.potencialSalida = self.potencialActivacionSalidas[0:1,:]
        
        self.Y =self.sigmoid(self.potencialSalida)
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

    def entrenar(self):
        for epoca in range (self.epocas):
            error_total =0
            for i,entrada in enumerate(self.arregloEntrenamiento):
                    
                #propagacion hacia adelante
                salida_calculada =self.propagacion(entrada)
                #calcular el error
                error =0.333*(self.arregloClase[i]-salida_calculada)
                print(f"  SALIDA ESPERADA {self.arregloClase[i]}    SALIDA OBTENIDA {salida_calculada}")
                error_total +=np.sum(error**2)
                
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


# Proporcionar los datos de entrenamiento y las clases
datos_entrenamiento = np.array([[1, 0, 1], [1, 1, 0],[0,1,1],[1,1,1],[0,0,0]])  # Agrega más ejemplos de entrenamiento
#clases = np.array([0, 1, 2,3,4])  # Agrega las clases correspondientes
clases = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]])
matrices_rgb = [dato.reshape(3, 1) for dato in datos_entrenamiento]



red_neuronal = MulticapaFinal(datos_entrenamiento, clases)

# Llamar al método de propagación con una entrada de ejemplo
entrada_ejemplo = np.array([0, 0, 0])

red_neuronal.entrenar()
# Imprimir los nuevos valores de pesos y umbrales
print("Nuevos pesos de salida:", red_neuronal.pesoSalidas)
print("Nuevos umbrales de salida:", red_neuronal.umbralNeuronaSalida)
print("Nuevos pesos de capa oculta:", red_neuronal.pesoOcultas)
print("Nuevos umbrales de capa oculta:", red_neuronal.umbralNeuronasOcultas)


entrada_prueba = np.array([0, 0, 0],)
salida_calculada = red_neuronal.propagacion(entrada_prueba)

clase_predicha = np.argmax(salida_calculada)
clase_real = 2  # La clase real correspondiente a la entrada de prueba
if clase_predicha == clase_real:
    print("La red neuronal ha clasificado correctamente la entrada.")
else:
    print(f"La red neuronal ha clasificado incorrectamente. Clase predicha: {clase_predicha}")
