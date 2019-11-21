import tkinter as Tk

def Calculer():
    a = float(e1.get())
    b = float(e2.get())
    c = float(e3.get())
    result = a+(a*b/c)/100
    lbl1.config(text = 'result '+str(result))

root = Tk.Tk()
e1 = Tk.Entry()
e2 = Tk.Entry()
e3 = Tk.Entry()
e1.pack()
e2.pack()
e3.pack()
b1 = Tk.Button(text='Calculer', command=Calculer)
b1.pack()
lbl1 = Tk.Label()
lbl1.pack()

root.mainloop()
