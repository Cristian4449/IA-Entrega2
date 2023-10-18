import numpy as np

class Multicapa():

    def __init__(self,datosEntrenamiento,datosClase):
        self.tasaAprendizaje=0.3
        self.precision =0.0000001
        self.epocas=100
        
        self.numeroEntradas=3
        self.capasOculta=1
        self.neuronasOcultas=3
        self.neuronaSalida=5
        
        self.umbralNeuronaSalida=np.ones((self.neuronaSalida,1)dtype=float)
        
        self.umbralNeuronasOcultas=np.ones((self.neuronasOcultas,1),dtype=float)
        
        np.random.seed(0)
        self.peso_1=np.random.rand(self.neuronasOcultas,self.numeroEntradas)
        self.peso_2=np.random.rand(self.neuronaSalida,self.neuronasOcultas)
        
        self.arregloEntrenamiento=np.transpose(datosEntrenamiento)
        self.arregloClase=datosClase
        
        #variables de inicializacion
        self.salidaEsperada=0 #Salida deseada en la iteracion actual
        self.errorRed=1#Error total de la red en un conjunto de las iteracion
        self.errorCuadratico=0
        self.error_previo=0 # error anterior
        self.errores=[]
        self.errorActual=np.zeros((len(self.arregloClase)))
        self.entradas=np.zeros((1,self.numeroEntradas))
        self.potencialActivacionOcultas=np.zeros((self.neuronasOcultas,1))#potencial de activacion en las neuronas ocultas
        self.funcionActivacionOculta=np.zeros((self.neuronasOcultas,1))
        self.Y=0.0#Potencial de activacion en la neurona de la salida
        self.salidaObtenida=0.0#Funcion de Activacion en neurona de salida
        self.epocasUsadas=0
        
        #Variables de retropropagacion
        self.errorReal=0
        self.deltaSalida=0.0
        self.deltaNeuronasOcultas=np.zeros((self.neuronasOcultas,1))#Deltas en neuronas ocultas
    
    def Sigmoide(self,datos):
        return 1/(1+np.exp(-datos))
    
    def derivadaSigmoide(self,datos):
        s=self.Sigmoide(datos)
        return s*(1-s)
    
    def Probar(self,x):
        respuesta=np.zeros((len(x),1))
        z=np.transpose(x)
        for salida in range(len(x)):
            self.entradas=z[:,salida]
            self.Propagar()
            respuesta[salida,:]=self.salidaObtenida
        return respuesta.tolist()
    
    
    
    
    def Softmax(self, x):
        exp_x = np.exp(x - np.max(x))  # Resta el máximo valor para mejorar la estabilidad numérica
        return exp_x / exp_x.sum(axis=0)


    '''
    def Propagar(self):
        #esta es la funcion de activacion de las primeras capas es lo mismo en el perceptron peso*entrada+Bias
        for i in range(self.neuronasOcultas):
            self.potencialActivacionOcultas[i,:]=np.dot(self.peso_1[i,:],self.entradas.T)+self.umbralNeuronasOcultas[i,:]
        
        for j in range(self.neuronasOcultas):
            self.funcionActivacionOculta[j,:]=self.Sigmoide(self.potencialActivacionOcultas[j,:])
        
        #Calcula el potencial para la neurona de salida
        self.Y=(np.dot(self.peso_2,self.funcionActivacionOculta)+self.umbralNeuronaSalida)
        self.salidaObtenida=self.Sigmoide(self.Y)
    '''
    def Propagar(self):
        #esta es la funcion de activacion de las primeras capas es lo mismo en el perceptron peso*entrada+Bias
        for i in range(self.neuronasOcultas):
            self.potencialActivacionOcultas[i,:]=np.dot(self.peso_1[i,:],self.entradas.T)+self.umbralNeuronasOcultas[i,:]
        
        for j in range(self.neuronasOcultas):
            self.funcionActivacionOculta[j,:]=self.Sigmoide(self.potencialActivacionOcultas[j,:])
        
        #Calcula el potencial para la neurona de salida
        self.Y = np.dot(self.peso_2, self.funcionActivacionOculta) + self.umbralNeuronaSalida
        self.salidaObtenida = self.Softmax(self.Y)
        
    def backPropagation(self):
        print("ENTRO A BACK PROPAGATION")
        #Para Saber si funciona XD
        self.errorReal=(self.salidaEsperada-self.salidaObtenida)
        #calcular salida
        self.deltaSalida=(self.derivadaSigmoide(self.Y)*self.errorReal)
        #ajustar peso2  o  sea el peso de la neurona salida no es tan dificil bobos hptas
        self.peso_2=self.peso_2 +(np.transpose(self.funcionActivacionOculta)*self.tasaAprendizaje*self.deltaSalida)
        #ajuste del umbral de salida o sea de la neurona final pap
        self.umbralNeuronaSalida=self.umbralNeuronaSalida + (self.tasaAprendizaje *self.deltaSalida)
        #calcular delta de las neuronas ocultas 
        self.deltaNeuronasOcultas = np.dot(np.transpose(self.peso_2), self.derivadaSigmoide(self.potencialActivacionOcultas)) * self.deltaSalida

        #Ajustar pesos de las primeras entradas
        for i in range(self.neuronasOcultas):
            self.peso_1[i,:]=self.peso_1[i,:]+((self.deltaNeuronasOcultas[i,:])*self.entradas*self.tasaAprendizaje)
        
        #Ajustar el umbral sesgo bias esa monda manito de las capas ocultas manito
        for i in range(self.neuronasOcultas):
            self.umbralNeuronasOcultas[i,:]=self.umbralNeuronasOcultas[i,:]+(self.tasaAprendizaje*self.deltaNeuronasOcultas[i,:])
    
    def Error(self):
        self.errorCuadratico=((1/len(self.arregloClase))*(sum(self.errorActual)))
        self.errorRed=self.errorCuadratico-self.error_previo
    
    def entrenar(self):
        print("entrenando...............")
        while(np.abs(self.errorRed)>self.precision):
            self.error_previo=self.errorCuadratico
            for i in range(len(self.arregloClase)):
                self.entradas=self.arregloEntrenamiento[:,i]#Señales de entrada por ciclo
                self.salidaEsperada=self.arregloClase[i]
                self.Propagar()
                self.backPropagation()
                self.Propagar()
                self.errorActual[i]=(0.3333)*((self.salidaEsperada-self.salidaObtenida)**2)
                print(f"SALIDA OBTENIDA     {self.salidaObtenida}   salida esperada      {self.salidaEsperada}")
            self.Error()
            self.epocasUsadas +=1
            if self.epocasUsadas>self.epocas:
                break
        print("entrenado")
    