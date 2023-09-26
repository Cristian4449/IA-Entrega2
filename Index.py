import tkinter as tk
import Menu as m

global variableDesicion
variableDesicion=tk.IntVar()

ventanaIndex=tk.Tk()
ventanaIndex.geometry("600x600")
ventanaIndex.title("INICIO")

def abrir():
    if(variableDesicion.get()==0):
        ventanaIndex.withdraw()
        m.ventanaMenu.deiconify()
        m.entrenamientoClasificacion(variableDesicion.get())
    elif(variableDesicion.get()==1):
        ventanaIndex.withdraw()
        m.mostrarVentanaEntrenamiento()
        m.obtenerClaseParaEntrenamiento(variableDesicion.get())
    
etiqueta = tk.Label(ventanaIndex,text="HOLA MUNDO",bg="green")
etiqueta.pack(fill=tk.X)
#command=lambda:saludo   permite sincronizar funciones.
boton1=tk.Button(ventanaIndex, text="Practica",command=lambda:variableDesicion.set(0), width=15 ,height=5)
boton1.place(x=270,y=150)

boton2=tk.Button(ventanaIndex, text="CLASIFICACION",command=lambda :variableDesicion.set(1), width=15 ,height=5)
boton2.place(x=270,y=250)


boton3 =tk.Button(ventanaIndex,text="GUARDAR",command=abrir, width=10 ,height=5)
boton3.place(x=500,y=500)

def mostrarVentanaInicio():
    ventanaIndex.deiconify()

def esconderVentanaMenu(event):
    m.ventanaMenu.withdraw()

ventanaIndex.bind("Escape",esconderVentanaMenu)

ventanaIndex.mainloop()

