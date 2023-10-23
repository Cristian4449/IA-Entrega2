#PAG ENTRENAMIENTO
import tkinter as tk
import cv2 as c
from PIL import Image, ImageTk
import numpy as np
import json
import os
from MulticapaFinal import MulticapaFinal
import serial
from time import sleep

#Inicializamos el puerto de serie a 9600 baud
ser = serial.Serial('COM5', 9600)
sleep(5)

ventanaEntrenamiento = tk.Tk()

ventanaEntrenamiento.geometry("600x600")
ventanaEntrenamiento.withdraw()
vectorClase=[]
vector=[]
ventanaMostrar =None

data_input=[]
def llamarEntrada():
    ventanaEntrenamiento.deiconify()

#Funcion para actualizar el video en tkinter
def actualizarVideo():
    global ventanaMostrar
    confirmarCaptura,ventanaMostrar= cap.read() #confirmarCaptura nos dice si la conexion funciono o no /TrueOrFalse,ventanaMostrar recibe tambien los datos de capRead()que son los pixeles que se mostraran del video
    if confirmarCaptura:
        #tenemos que convertir el frame de OpenCV de BGR A RGB, para que sea compatible con nuestros datos.
        ventanaMostrar=c.cvtColor(ventanaMostrar,c.COLOR_BGR2RGB)
        ventanaMostrar=c.rotate(ventanaMostrar,c.ROTATE_90_CLOCKWISE)
        ventanaMostrar=c.resize(ventanaMostrar,(600,600))
        #lo que hacemos aqui es para que el frame de openCV se adapte ala ventan tkinter
        foto=ImageTk.PhotoImage(image=Image.fromarray(ventanaMostrar))
        #Actualizamos la etiqueta con la imagen ya adaptada para tkinter
        etiquetaVideo.config(image=foto)
        etiquetaVideo.foto = foto
        etiquetaVideo.after(10,actualizarVideo)#cada 0.01 segundos se actualiza el video o el label

def capturarArea(event):
    global data_input
    global vectorClase
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

    suma_r=[]
    suma_g=[]
    suma_b=[]
    for fila in area:
        for r,g,b in fila:
            suma_r.append(r/255)
            suma_g.append(g/255)
            suma_b.append(b/255)
            

    totalPixeles= 49
    suma_r=np.array(suma_r)
    suma_g=np.array(suma_g)
    suma_b=np.array(suma_b)
    
    promedioR=np.mean(suma_r)    
    promedioG=np.mean(suma_g)    
    promedioB=np.mean(suma_b)    
    

    print(f"EL PROMEDIO QUE DA DE LOS COLORES ES :    R:   {promedioR}   G: {promedioG}  B: {promedioB}")
    print(f"VARIABLE CLASIFICAR ES {variableClasificar}")
    if(variableClasificar==0):
        guardarDatosParaEntrenar(promedioR,promedioG,promedioB)
    
    elif(variableClasificar==1):
        print(f"BUENO ENTRANDO A CLASIFICACION")
        arregloClasificacion =np.array([[promedioR,promedioG,promedioB]])
        
        resultado= multicapa.clasificar(arregloClasificacion)
        if(np.argmax(resultado) == 0):
            nombre = "papa"
        elif(np.argmax(resultado) == 1):
            nombre = "uva"
        elif(np.argmax(resultado) == 2):
            nombre = "limón"
        elif(np.argmax(resultado) == 3):
            nombre = "Fruta x1"
        elif(np.argmax(resultado) == 4):
            nombre = "Fruta x2"
        
        
        print(f"{resultado}   EL RESULTADO ESSSSSS     {nombre}")
        
        entrada = np.argmax(resultado)
        ser.write(str(entrada).encode())
        
# Inicializa la camara o la fuente de video
cap = c.VideoCapture('http://192.168.100.20:4747/video')


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




arregloColor=[]
clase=[]
claseArreglo=[]
multicapa=MulticapaFinal(arregloColor,clase)

def entrenarPesosMultiCapa(event):
    global arregloColor,clase,multicapa,claseArreglo
    with open('Frutas.json','r') as archivo_json:
        datosCargados=json.load(archivo_json)
        r=[datos['R'] for datos in datosCargados]
        g=[datos['G'] for datos in datosCargados]
        b=[datos['B'] for datos in datosCargados]
        clase=[datos['Clase'] for datos in datosCargados]

    for valorR,valorG,valorB,numero in zip(r,g,b,clase):
        arregloColor.append([valorR,valorG,valorB])
        if numero==0:
            claseArreglo.append([1,0,0,0,0])
        elif numero==1:
            claseArreglo.append([0,1,0,0,0])
        elif numero==2:
            claseArreglo.append([0,0,1,0,0])
        elif numero==3:
            claseArreglo.append([0,0,0,1,0])
        elif numero==4:
            claseArreglo.append([0,0,0,0,1])
            
    arregloColor = np.array(arregloColor)
    claseArreglo=np.array(claseArreglo)
    
    multicapa=MulticapaFinal(arregloColor,claseArreglo)
    multicapa.entrenar()

    
#ventanaEntrenamiento.bind("<space>",entrenarPesosMultiCapa)
ventanaEntrenamiento.bind("<space>",entrenarPesosMultiCapa)

variableClasificar=None

def obtenerClase(x):
    global variableClasificar
    variableClasificar=x
    