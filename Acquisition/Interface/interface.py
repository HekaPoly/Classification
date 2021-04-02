from tkinter import *
import serial
import time
import numpy as np
import queue
import threading
import os
import matplotlib.pyplot as plt
import pathlib

# from tempfile import TemporaryFile

# outfile = TemporaryFile()

kSizePacket = 100
kMesurePerSecond = 4000 # Hertz (For each electrodes)

instant = 0
StopSerial = False

comPort = "COM4"

def process_serial_buffer(q, name, movement_class, acq_number, n_electrodes, acq_time, n_mesures, file_path):

    global instant
    global StopSerial

    index = np.zeros(n_electrodes) #To keep track of the index of each electrod
    pin_number = 0
    emg_data = np.zeros((n_electrodes,kMesurePerSecond * acq_time))
    n = 0

    while True:
        low = q.get()
        high = q.get()

        # print("low = " + str(low))
        # print("high = " + str(high))
        
        if(low == 0xFF and high == 0xFF):
                pin_number = int(q.get())
                print("pin number is " + str(pin_number))
        else :
            value = low + (high << 8)
            # print("value = " + str(value))

            i = int(index[pin_number-1])
            
            # print("i = " + str(i))
            emg_data[pin_number-1][i] = value
            index[pin_number-1] = index[pin_number-1] + 1
            n = n + 1

        if n==n_mesures:
            print(emg_data)
            print("time is " + str(instant))

            # np.savetxt(file_name + ".txt", emg_data)
            np.save(file_path, emg_data)

            StopSerial = True

            for j in range(0, pin_number-1):
                index[j] = 0

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

        # Paramètres
        self.name = "SubjectName"
        self.acquisition_time = 1
        self.movement_class = "MovementClass"
        self.acquisition_number = 1
        self.n_electrodes = 1
        self.n_mesures = 0
        self.file_name = ""
        self.file_path = ""

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
        self.e5 = Entry(self.left_frame, font=("Courrier", 20), bg='white', fg='#6E6C6C')
        self.e5.grid(row=11)

        self.start_button = Button(self.right_frame, text="Start", font=("Courrier", 25), bg='#F90A0A', fg='#6E6C6C',
                           command=self.acquisition)
        self.start_button.grid(row=1)
        self.start_button.config (state = DISABLED)

        self.init_fields()

    def create_widgets(self):
        self.create_title()
        self.create_subtitle()
        # self.create_start_button()
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
        label_subtitle_1 = Label(self.left_frame, text="Acquisition time:", font=("Courrier", 15), bg='#6E6C6C',
                                 fg='white')
        label_subtitle_1.grid(row=2)

        label_subtitle_2 = Label(self.left_frame, text="Movement Class:", font=("Courrier", 15), bg='#6E6C6C',
                                 fg='white')
        label_subtitle_2.grid(row=4)

        label_subtitle_3 = Label(self.left_frame, text="Acquisition Number:", font=("Courrier", 15), bg='#6E6C6C',
                                 fg='white')
        label_subtitle_3.grid(row=6)

        label_subtitle_4 = Label(self.left_frame, text="Number of electrodes:", font=("Courrier", 15), bg='#6E6C6C',
                                 fg='white')
        label_subtitle_4.grid(row=8)

        label_subtitle_5 = Label(self.left_frame, text="Name:", font=("Courrier", 15), bg='#6E6C6C',
                                 fg='white')
        label_subtitle_5.grid(row=10)

    # def create_start_button(self):
    #     start_button = Button(self.right_frame, text="Start", font=("Courrier", 25), bg='#F90A0A', fg='#6E6C6C',
    #                        command=self.acquisition)
    #     start_button.grid(row=1)
    #     start_button.config (state = DISABLED)

    def create_load_button(self):
        load_button = Button(self.right_frame, text="Load", font=("Courrier", 25), bg='white', fg='#6E6C6C',
                           command=self.load)
        load_button.grid(row=4)

    def create_init_button(self):
        load_button = Button(self.left_frame, text="Init", font=("Courrier", 25), bg='white', fg='#6E6C6C',
                           command=self.initate_aquisition)
        load_button.grid(row=13)

    def create_empty_label(self):
        label_empty_1 = Label(self.left_frame, bg='#6E6C6C')
        label_empty_1.grid(row=1)
        label_empty_2 = Label(self.left_frame, bg='#6E6C6C')
        label_empty_2.grid(row=12)

        label_empty_3 = Label(self.right_frame, bg='#6E6C6C')
        label_empty_3.grid(row=2)

    def init_fields(self):
        self.e1.insert(0,self.acquisition_time)
        self.e2.insert(0,self.movement_class)
        self.e3.insert(0,self.acquisition_number)
        self.e4.insert(0,self.n_electrodes)
        self.e5.insert(0, self.name)

    def initate_aquisition(self):
        global StopSerial

        self.acquisition_time = int(self.e1.get())
        self.movement_class = self.e2.get()
        self.acquisition_number = int(self.e3.get())
        self.nElectrodes = int(self.e4.get())
        self.name = self.e5.get()

        self.n_mesures = kMesurePerSecond * self.n_electrodes * self.acquisition_time
        self.file_name = self.movement_class + "_" + str(self.acquisition_time) + "s_" + str(kMesurePerSecond) + "Hz_" + str(self.acquisition_number)
        self.file_path = os.path.join("Data", self.name, self.file_name + ".npy")

        # Make sure folder exist
        name_path = os.path.join(pathlib.Path().absolute(), "Data")
        list_subfolders = [f.name for f in os.scandir(name_path) if f.is_dir()]
        if self.name not in list_subfolders:
            os.mkdir("Data\\"+self.name)

        StopSerial = False

        print("Acquisition time: ", self.acquisition_time)
        print("Movement class: ", self.movement_class)
        print("Acquisition number: ", self.acquisition_number)
        print("Number of electrodes: ", self.n_electrodes)
        print("Number of mesures: ", self.n_mesures)
        print("Name: ", self.name)
        print("Path: ", self.file_path)

        self.start_button.config (state = NORMAL,bg='#16F90A')

    def acquisition(self):
        print("Start acquisition")
        self.start_button.config (state = DISABLED,bg='#F90A0A')
        port_open = False
        while not port_open:
            try:
                ser = serial.Serial(comPort, timeout=None, baudrate=115200, xonxoff=False, rtscts=False, dsrdtr=False)
                ser.flushInput()
                ser.flushOutput()
                port_open = True
                print("port is open")
            except:
                print("Can't connect to port")
                exit(1)

        q = queue.Queue()
        consumer = threading.Thread(target=process_serial_buffer, args=(q, 
                                                                        self.name, 
                                                                        self.movement_class, 
                                                                        self.acquisition_number, 
                                                                        self.n_electrodes,
                                                                        self.acquisition_time,
                                                                        self.n_mesures,
                                                                        self.file_path))
        
        consumer.start()


        ser.write(bytearray('#','ascii'))

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
                    ser.write(bytearray('!','ascii'))
                    # Update acquisition number
                    self.e3.delete(0, 'end')
                    self.acquisition_number = self.acquisition_number + 1
                    self.e3.insert(0,self.acquisition_number)
                    break

            print("counter = " + str(counter))

        except KeyboardInterrupt:
            print('interrupted!')

    # Afficher les resultats
    def load(self):
        data = np.load(self.file_path)
        position = 211

        if self.n_electrodes > 2 :
            position = 221

        if self.n_electrodes > 4 :
            position = 221

        for k in range(0,self.n_electrodes):
            if k < 8:
                plt.subplot(position)
                plt.plot(data[k])
                position = position + 1
            else :
                print("Not enough space on screen plot mesures from elecrode n°" + str(k+1))

        # plt.subplot(241)
        # plt.plot(data[0])
        # plt.subplot(242)
        # plt.plot(data[1])
        # plt.subplot(243)
        # plt.plot(data[2])
        # plt.subplot(244)
        # plt.plot(data[3])
        # plt.subplot(245)
        # plt.plot(data[4])
        # plt.subplot(246)
        # plt.plot(data[5])
        # plt.subplot(247)
        # plt.plot(data[6])
        plt.show()
        print("not ready")

# afficher
interface = MyInterface()
interface.window.mainloop()
