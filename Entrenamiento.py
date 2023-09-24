#PAG ENTRENAMIENTO
import tkinter as tk
import cv2 as c
from PIL import Image, ImageTk
import numpy as np
import json
import os

ventanaEntrenamiento = tk.Tk()

ventanaEntrenamiento.geometry("600x600")
ventanaEntrenamiento.withdraw()

vector=[]
ventanaMostrar =None


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

    guardarDatosParaEntrenar(promedioR,promedioG,promedioB)

# Inicializa la camara o la fuente de video
cap = c.VideoCapture('http://192.168.18.9:4747/video')


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