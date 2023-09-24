#PAG PRACTICA
import tkinter as tk
import Entrenamiento as e

ventanaMenu = tk.Tk()
ventanaMenu.withdraw()
ventanaMenu.geometry("600x600")

variableClasificacion=None

valorFruta = tk.IntVar()
papa = tk.Button(ventanaMenu, text="PAPA", width=20, height=5,command=lambda : valorFruta.set(0))
uva = tk.Button(ventanaMenu, text="UVA", width=20, height=5,command=lambda : valorFruta.set(1))
limon = tk.Button(ventanaMenu, text="LIMON", width=20, height=5,command=lambda : valorFruta.set(2))
frutaMaestro=tk.Button(ventanaMenu,text="FRUTA NUEVA 1",width=20,height=5,command=lambda:valorFruta.set(3))
frutaMaestro2=tk.Button(ventanaMenu,text="FRUTA NUEVA 2",width=20,height=5,command=lambda:valorFruta.set(4))

def caso():
    ventanaMenu.withdraw()
    e.llamarEntrada()
    e.extraerDatos(variableClasificacion,valorFruta.get())

guardar = tk.Button(ventanaMenu,text="GUARDAR FRUTA",width=10,height=3,command=caso)

guardar.place(x=500,y=500)
papa.place(x=230, y=50)
uva.place(x=230, y=150)
limon.place(x=230, y=250)
frutaMaestro.place(x=230,y=350)
frutaMaestro2.place(x=230,y=450)


def entrenamientoClasificacion(entrada):
    global variableClasificacion
    variableClasificacion=entrada

def retrocederInicio(event):
    from Index import mostrarVentanaInicio
    mostrarVentanaInicio()

ventanaMenu.bind("<Escape>",retrocederInicio)

def mostrarVentana():
    ventanaMenu.deiconify()
    e.ventanaEntrenamiento.withdraw()