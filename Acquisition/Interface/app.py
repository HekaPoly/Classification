from tkinter import *
import serial
import time
import numpy as np
import matplotlib.pyplot as plt
from tempfile import TemporaryFile

outfile = TemporaryFile()

AcquisitionTime = 0
movement_class = 0
acquisition_number = 0
Name = 'NoName'
valeurs=[]
temps=[]

class MyApp:

    def __init__(self):
        self.window = Tk()
        self.window.title("Signal Acquisition Interface")
        self.window.geometry("720x480")
        self.window.minsize(480, 360)
        self.window.iconbitmap("logo.ico")
        self.window.config(background='#6E6C6C')

        # initialization des composants
        self.left_frame = Frame(self.window, bg='#6E6C6C')
        self.right_frame = Frame(self.window, bg='#6E6C6C')

        # creation des composants
        self.create_widgets()

        # empaquetage
        # self.left_frame.pack(expand=YES)
        self.left_frame.grid(row=0, column=0, sticky=N)
        self.right_frame.grid(row=0, column=1, sticky=N)

        # Variable d'aquisition
        self.e1 = Entry(self.left_frame, font=("Courrier", 20), bg='white', fg='#6E6C6C')
        self.e1.grid(row=3)
        self.e2 = Entry(self.left_frame, font=("Courrier", 20), bg='white', fg='#6E6C6C')
        self.e2.grid(row=5)
        self.e3 = Entry(self.left_frame, font=("Courrier", 20), bg='white', fg='#6E6C6C')
        self.e3.grid(row=7)
        self.e4 = Entry(self.left_frame, font=("Courrier", 20), bg='white', fg='#6E6C6C')
        self.e4.grid(row=9)

        self.init_fields()


    def create_widgets(self):
        self.create_title()
        self.create_subtitle()
        self.create_start_button()
        self.create_init_button()
        self.create_load_button()
        self.create_empty_label()

    def create_title(self):
        label_title_1 = Label(self.right_frame, text="Press start to launch the aquisition", font=("Courrier", 18),
                              bg='#6E6C6C',
                              fg='white')
        label_title_1.grid(row=0)

        label_title_2 = Label(self.left_frame, text="Enter the parameters", font=("Courrier", 18), bg='#6E6C6C',
                              fg='white')
        label_title_2.grid(row=0)

        label_title_3 = Label(self.right_frame, text="Press load to display the data", font=("Courrier", 18),
                              bg='#6E6C6C',
                              fg='white')
        label_title_3.grid(row=3)

    def create_subtitle(self):
        label_subtitle_1 = Label(self.left_frame, text="Acquisition time :", font=("Courrier", 15), bg='#6E6C6C',
                                 fg='white')
        label_subtitle_1.grid(row=2)

        label_subtitle_2 = Label(self.left_frame, text="Classe de mouvement :", font=("Courrier", 15), bg='#6E6C6C',
                                 fg='white')
        label_subtitle_2.grid(row=4)

        label_subtitle_3 = Label(self.left_frame, text="Numéro d'acquisition :", font=("Courrier", 15), bg='#6E6C6C',
                                 fg='white')
        label_subtitle_3.grid(row=6)

        label_subtitle_4 = Label(self.left_frame, text="Nom :", font=("Courrier", 15), bg='#6E6C6C',
                                 fg='white')
        label_subtitle_4.grid(row=8)

    def create_start_button(self):
        start_button = Button(self.right_frame, text="Start", font=("Courrier", 25), bg='white', fg='#6E6C6C',
                           command=self.acquisition)
        start_button.grid(row=1)

    def create_load_button(self):
        load_button = Button(self.right_frame, text="Load", font=("Courrier", 25), bg='white', fg='#6E6C6C',
                           command=self.load)
        load_button.grid(row=4)

    def create_init_button(self):
        load_button = Button(self.left_frame, text="Init", font=("Courrier", 25), bg='white', fg='#6E6C6C',
                           command=self.initate_aquisition)
        load_button.grid(row=11)

    def create_empty_label(self):
        label_empty_1 = Label(self.left_frame, bg='#6E6C6C')
        label_empty_1.grid(row=1)
        label_empty_2 = Label(self.left_frame, bg='#6E6C6C')
        label_empty_2.grid(row=10)
        label_empty_3 = Label(self.right_frame, bg='#6E6C6C')
        label_empty_3.grid(row=2)



    def init_fields(self):
        self.e1.insert(0,AcquisitionTime)
        self.e2.insert(0,movement_class)
        self.e3.insert(0,acquisition_number)
        self.e4.insert(0,Name)

    def initate_aquisition(self):
        global AcquisitionTime
        global movement_class
        global acquisition_number
        global Name

        AcquisitionTime = float(self.e1.get())
        movement_class = int(self.e2.get())
        acquisition_number = int(self.e3.get())
        Name = self.e4.get()

        print("Acquisition time = ", AcquisitionTime)
        print("Movement class = ", movement_class)
        print("Acquisition number = ", acquisition_number)
        print("Name = ", Name)

    def acquisition(self):
        global AcquisitionTime
        global valeurs

        # Start communication with teensy
        teensy = serial.Serial('COM6', 115200, timeout=.1)
        time.sleep(1)  # give the connection a second to settle
        teensy.write(1)

        start=time.time()
        instant = 0
        # NewElectrode = false

        while instant < AcquisitionTime:
            data = teensy.readline()
            if data:
                # if("#" in data):
                #     NewElectrode = true
                mesure= data #float(data)
                valeurs.append(mesure)
                temps.append(instant)
                print(mesure,instant)
            instant=time.time()-start

        results = np.concatenate((valeurs,temps),axis=0)
        # np.save('test1.npy', results)

    def load(self):

        data = np.load("elbow_extension_3s_2000Hz_1.npy")
        plt.subplot(241)
        plt.plot(data[0])
        plt.subplot(242)
        plt.plot(data[1])
        plt.subplot(243)
        plt.plot(data[2])
        plt.subplot(244)
        plt.plot(data[3])
        plt.subplot(245)
        plt.plot(data[4])
        plt.subplot(246)
        plt.plot(data[5])
        plt.subplot(247)
        plt.plot(data[6])
        plt.show()
        print("not ready")

# afficher
app = MyApp()
app.window.mainloop()
