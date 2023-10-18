#PAG ENTRENAMIENTO
import tkinter as tk
import cv2 as c
from PIL import Image, ImageTk
import numpy as np
import json
import os
from Perceptron import PerceptronProfe
from Multicapa import Multicapa
from sklearn.preprocessing import OneHotEncoder

ventanaEntrenamiento = tk.Tk()

ventanaEntrenamiento.geometry("600x600")
ventanaEntrenamiento.withdraw()

vector=[]
ventanaMostrar =None

p0 = PerceptronProfe(3)
p1 = PerceptronProfe(3)
p2= PerceptronProfe(3)
p3= PerceptronProfe(3)
p4= PerceptronProfe(3)

def llamarEntrada():
    ventanaEntrenamiento.deiconify()

#Funcion para actualizar el video en tkinter
def actualizarVideo():
    global ventanaMostrar
    confirmarCaptura,ventanaMostrar= cap.read() #confirmarCaptura nos dice si la conexion funciono o no /TrueOrFalse,ventanaMostrar recibe tambien los datos de capRead()que son los pixeles que se mostraran del video
    if confirmarCaptura:
        #tenemos que convertir el frame de OpenCV de BGR A RGB, para que sea compatible con nuestros datos.
        ventanaMostrar=c.cvtColor(ventanaMostrar,c.COLOR_BGR2RGB)
        ventanaMostrar=c.resize(ventanaMostrar,(600,600))
        #lo que hacemos aqui es para que el frame de openCV se adapte ala ventan tkinter
        foto=ImageTk.PhotoImage(image=Image.fromarray(ventanaMostrar))
        #Actualizamos la etiqueta con la imagen ya adaptada para tkinter
        etiquetaVideo.config(image=foto)
        etiquetaVideo.foto = foto
        etiquetaVideo.after(10,actualizarVideo)#cada 0.01 segundos se actualiza el video o el label

def capturarArea(event):
    global ventanaMostrar
    global vector
    global variableClasificar
    x,y = event.x, event.y
    area= []
    for ancho in range(x-3,x+4):# o sea que empiece en la posicion X -3 y avance hasta x+4
        fila=[]
        for alto in range(y-3,y+4):
            try:
                imagen_pillow= Image.fromarray(ventanaMostrar)
                obtenerPixel =imagen_pillow.getpixel((ancho,alto))
                fila.append(obtenerPixel)
            except IndexError:
                fila.append((255,255,255))#pos si se sale de la imagen recoja datos en blanco pero es imposible aparentementeeeeeeeeeeee
        area.append(fila)# aqui la lista fila se agrega a la lista area
    suma_r=0
    suma_g=0
    suma_b=0
    '''
    for fila in area:
       print(fila)#para mostrar los datos 
    '''
    for fila in area:
        for r,g,b in fila:
            suma_r+=r
            suma_g+=g
            suma_b+=b
            

    totalPixeles= 49
    
   
    promedioR =suma_r//totalPixeles    
    promedioG =suma_g//totalPixeles    
    promedioB =suma_b//totalPixeles    

    print(f"EL PROMEDIO QUE DA DE LOS COLORES ES :    R:   {promedioR}   G: {promedioG}  B: {promedioB}")
    print(f"VARIABLE CLASIFICAR ES {variableClasificar}")
    if(variableClasificar==0):
        print(f'EL VECTOR CERO ES {variableClasificar} modo PRACTICA')
        guardarDatosParaEntrenar(promedioR,promedioG,promedioB)
    elif(variableClasificar==1):
        
        print(f'EL VECTOR CERO ES {variableClasificar} modo clasificacion')
        arreglo_prueba = np.array([[promedioR,promedioG,promedioB]])
        
        resultado = multicapa.Probar(arreglo_prueba)
        print(f' EL RESULTADO ESSSSSS       {resultado} ')
        print(np.round(resultado))
        
    
        
# Inicializa la camara o la fuente de video
cap = c.VideoCapture('http://192.168.1.4:4747/video')


etiquetaVideo= tk.Label(ventanaEntrenamiento)#papi se supone que ya saben como es un label

etiquetaVideo.pack()#el pack hace que automaticamente la etiqueta se posicione por defecto en la ventana

actualizarVideo()

etiquetaVideo.bind("<Button-1>",capturarArea)

def guardarDatosParaEntrenar(r,g,b):
    datos_json=[]
    if os.path.exists('Frutas.json'):
        with open('Frutas.json','r') as archivo_json:
            datos_json=json.load(archivo_json)
            datos_json.append({'R':r,'G':g,'B':b,'Clase':vector[1]})
    with open('Frutas.json','w')as archivo_json:
        json.dump(datos_json,archivo_json,indent=4)
        print(f"SE GUARDARON LOS DATOS PARA LA CLASE  {vector[1]}")

def extraerDatos(clasificacion,fruta):
    global vector
    vector=clasificacion,fruta

def retrocederMenu(event):
    from Menu import mostrarVentana
    mostrarVentana()
    
ventanaEntrenamiento.bind("<Escape>",retrocederMenu)

def entrenarSacarPesos(event=None):
    global p0,p1,p2,p3,p4
    vector0R,vector0G,vector0B,vector1R,vector1G,vector1B,vector2R,vector2G,vector2B,vector3R,vector3G,vector3B,vector4R,vector4G,vector4B=([] for _ in range(15))
    vClase0,vClase1,vClase2,vClase3,vClase4=([] for _ in range(5))
    with open('Frutas.json','r') as archivo_json:
        datosCargados=json.load(archivo_json)
    for datos in datosCargados:
        if(datos['Clase']==0):
            vector0R.append(datos['R'])
            vector0G.append(datos['G'])
            vector0B.append(datos['B'])
            vClase0.append(datos['Clase'])
        elif(datos['Clase']==1):
            vector1R.append(datos['R'])
            vector1G.append(datos['G'])
            vector1B.append(datos['B'])
            vClase1.append(datos['Clase'])
        elif(datos['Clase']==2):
            vector2R.append(datos['R'])
            vector2G.append(datos['G'])
            vector2B.append(datos['B'])
            vClase2.append(datos['Clase'])
        elif(datos['Clase']==3):
            vector3R.append(datos['R'])
            vector3G.append(datos['G'])
            vector3B.append(datos['B'])
            vClase3.append(datos['Clase'])
        elif(datos['Clase']==4):
            vector4R.append(datos['R'])
            vector4G.append(datos['G'])
            vector4B.append(datos['B'])
            vClase4.append(datos['Clase'])
    vector0R=np.array(vector0R)
    vector0G=np.array(vector0G)
    vector0B=np.array(vector0B)
    vector1R=np.array(vector1R)
    vector1G=np.array(vector1G)
    vector1B=np.array(vector1B)
    vector2R=np.array(vector2R)
    vector2G=np.array(vector2G)
    vector2B=np.array(vector2B)
    vector3R=np.array(vector3R)
    vector3G=np.array(vector3G)
    vector3B=np.array(vector3B)
    vector4R=np.array(vector4R)
    vector4G=np.array(vector4G)
    vector4B=np.array(vector4B)
    
    print(p0.peso1)
    print(p0.peso2)
    print(p0.peso3)
    
    p0.entrenador(vector0R,vector0G,vector0B,vClase0)
    p1.entrenador(vector1R,vector1G,vector1B,vClase1)
    p2.entrenador(vector2R,vector2G,vector2B,vClase2)
    p3.entrenador(vector3R,vector3G,vector3B,vClase3)
    p4.entrenador(vector4R,vector4G,vector4B,vClase4)
    print(p0.peso1)
    print(p0.peso2)
    print(p0.peso3)


arregloColor=[]
clase=[]
multicapa=Multicapa(arregloColor,clase)

def entrenarPesosMultiCapa(event):
     global arregloColor,clase,multicapa
     with open('Frutas.json','r') as archivo_json:
        datosCargados=json.load(archivo_json)
        r=[datos['R'] for datos in datosCargados]
        g=[datos['G'] for datos in datosCargados]
        b=[datos['B'] for datos in datosCargados]
        clase=[datos['Clase'] for datos in datosCargados]
        #arregloColor=[r,g,b]
        arregloColor=list(zip(r,g,b))
        encoder = OneHotEncoder(sparse=False, categories='auto')
        etiquetas_one_hot = encoder.fit_transform(np.array(clase).reshape(-1, 1))
        multicapa=Multicapa(arregloColor,etiquetas_one_hot)
        multicapa.entrenar()

def calcularColor(x):
    resultado = multicapa.Probar(x)
    print(f' EL RESULTADO ESSSSSS       {resultado} ')
    print(np.round(resultado))
    
ventanaEntrenamiento.bind("<space>",entrenarPesosMultiCapa)

variableClasificar=None

def obtenerClase(x):
    global variableClasificar
    variableClasificar=x
    