from tkinter import *
import serial
import time
import numpy as np
import queue
import threading
import os
import matplotlib.pyplot as plt
from tempfile import TemporaryFile

outfile = TemporaryFile()

kAcquisitionTime = 1
movement_class = "TEST"
acquisition_number = 1
kNElectrodes = 9

kSizePacket = 100
kMesurePerSecond = 2000 # Hertz (For each electrodes)

name = "noName"

instant = 0
StopSerial = False

def process_serial_buffer(q):

    global instant
    global name
    global kNumOfMesures
    global StopSerial

    index= np.zeros(9)
    pin_number = 0
    emg_data = np.zeros((kNElectrodes,kMesurePerSecond * kAcquisitionTime))
    n = 0

    while True:
        low = q.get()
        # print("low = " + str(low))
        high = q.get()
        # print("high = " + str(high))
        if(low == 0 and high == 0):
                pin_number = int(q.get())
                # print("pin number is " + str(pin_number))
        else :
            value = low + (high << 8)
            # print("value = " + str(value))
            i = int(index[pin_number-1])
            # print("i = " + str(i))
            emg_data[pin_number-1][i] = value
            index[pin_number-1] = index[pin_number-1] + 1
            n = n + 1

        if n==kNumOfMesures:
            print(emg_data)
            print("time is " + str(instant))
            name = movement_class + "_" + str(kAcquisitionTime) + "s_" + str(kMesurePerSecond) + "Hz_" + str(acquisition_number)
            np.savetxt(name + ".txt", emg_data)
            np.save(name + ".npy", emg_data)

            StopSerial = True

            break
            # os._exit(0)

class MyInterface:

    def __init__(self):
        self.window = Tk()
        self.window.title("Signal Acquisition Interface")
        self.window.geometry("720x480")
        self.window.minsize(480, 360)
        self.window.iconbitmap("logo.ico")
        self.window.config(background='#6E6C6C')

        # initialisation des composants
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

        label_subtitle_2 = Label(self.left_frame, text="Mouvementclass :", font=("Courrier", 15), bg='#6E6C6C',
                                 fg='white')
        label_subtitle_2.grid(row=4)

        label_subtitle_3 = Label(self.left_frame, text="Acquisition Number:", font=("Courrier", 15), bg='#6E6C6C',
                                 fg='white')
        label_subtitle_3.grid(row=6)

        label_subtitle_4 = Label(self.left_frame, text="Number of electrodes :", font=("Courrier", 15), bg='#6E6C6C',
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
        self.e1.insert(0,kAcquisitionTime)
        self.e2.insert(0,movement_class)
        self.e3.insert(0,acquisition_number)
        self.e4.insert(0,kNElectrodes)

    def initate_aquisition(self):
        global kAcquisitionTime
        global movement_class
        global acquisition_number
        global kNElectrodes
        global kNumOfMesures

        kAcquisitionTime = int(self.e1.get())
        movement_class = self.e2.get()
        acquisition_number = int(self.e3.get())
        kNElectrodes = int(self.e4.get())
        kNumOfMesures = kMesurePerSecond * kNElectrodes * kAcquisitionTime

        print("Acquisition time = ", kAcquisitionTime)
        print("Movement class = ", movement_class)
        print("Acquisition number = ", acquisition_number)
        print("Number of electrodes = ", kNElectrodes)
        print("Number of mesures = ", kNumOfMesures)


    def acquisition(self):

        port_open = False
        while not port_open:
            try:
                ser = serial.Serial("COM6", timeout=None, baudrate=115200, xonxoff=False, rtscts=False, dsrdtr=False)
                ser.flushInput()
                ser.flushOutput()
                port_open = True
                if port_open :
                    print("port is open")
            except:
                pass

        q = queue.Queue()
        consumer = threading.Thread(target=process_serial_buffer, args=(q,))
        consumer.start()


        ser.write(1)

        global instant
        global StopSerial

        start_time = False
        start = 0
        counter = 0

        try:
            while True:
                bytesToRead = ser.inWaiting()
                if bytesToRead != 0:
                    if start_time == False:
                        start=time.time()
                        start_time = True
                        # print("start = " + str(start))
                    data = ser.read(bytesToRead)
                    # print(data)
                    for sample in data:
                        q.put(sample)
                        # counter = counter +1
                        # if counter == 2*kNumOfMesures + 3*kNumOfMesures/100:
                        #     print("time is " + str(instant) + " when counter is at" + str(counter))
                if start_time:
                    instant=time.time()-start

                if StopSerial:
                    print("Acquisition completed")
                    break

            print("counter = " + str(counter))

        except KeyboardInterrupt:
            print('interrupted!')


    # Afficher les resultats
    def load(self):

        global name

        data = np.load(name + ".npy")
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
interface = MyInterface()
interface.window.mainloop()
