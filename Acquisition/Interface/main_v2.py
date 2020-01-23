import serial
import numpy as np
import queue
import threading
import os
import math
import time

kNElectrodes = 9
kSizePacket = 100
kMesurePerSecond = 2000 # Hertz (For each electrodes)
kAcquisitionTime = 1
kNumOfMesures = kMesurePerSecond * kNElectrodes * kAcquisitionTime
print(kNumOfMesures)
movement_class = "TEST"
acquisition_number = "2"
instant = 0

def process_serial_buffer(q):

    global instant
    index= np.zeros(9)
    pin_number = 0
    emg_data = np.zeros((kNElectrodes,kMesurePerSecond * kAcquisitionTime))
    processed_packets = 0
    n = 0

    while True:
        low = q.get()
        # print("low = " + str(low))
        high = q.get()
        # print("high = " + str(high))
        if(low == 0 and high == 0):
                pin_number = int(q.get())
                # print("pin number is " + str(pin_number))
                # i=emg_data[pin_number].size
        else :
            value = low + (high << 8)
            # print("value = " + str(value))
            i = int(index[pin_number-1])
            # print("i = " + str(i))
            emg_data[pin_number-1][i] = value
            # emg_data.append(value)
            index[pin_number-1] = index[pin_number-1] + 1
            n = n + 1

            # if (n==kNumOfMesures): # kSizePacket):
            #     processed_packets = processed_packets + 1
            #     print("processed_packets = "+ str(processed_packets))
            #     # n = 0

        if n==kNumOfMesures:
            print(emg_data)
            print("time is " + str(instant))
            np.savetxt(movement_class + "_" + str(kAcquisitionTime) + "s_" + str(kMesurePerSecond) + "Hz_" + acquisition_number + ".txt", emg_data)
            np.save(movement_class + "_" + str(kAcquisitionTime) + "s_" + str(kMesurePerSecond) + "Hz_" + acquisition_number + ".npy", emg_data)
            os._exit(0)


def main():
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

    AcquisitionTime = 40

    ser.write(1)

    global instant
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
                    counter = counter +1
                    if counter == kNumOfMesures + 3*kNumOfMesures/100:
                        print("time is " + str(instant) + " when counter is at" + str(counter))
            if start_time:
                instant=time.time()-start

        print("counter = " + str(counter))
    except KeyboardInterrupt:
        print('interrupted!')
    # os._exit(0)


if __name__ == "__main__":
    main()
