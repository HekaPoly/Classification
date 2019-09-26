import serial
import numpy as np
import queue
import threading
import os
import math

kNElectrodes = 7
kSizePacket = 350
kNSamplesPerPacket = int(kSizePacket / 2)
kSamplePerSecond = 2000
kAcquisitionTime = 30
kNumPacketsToAcquire = math.ceil(kSamplePerSecond * kAcquisitionTime / kNSamplesPerPacket) * kNElectrodes

movement_class = "freestyle"
acquisition_number = "3"


def process_serial_buffer(q):
    electrode = 0
    processed_packets = 0
    emg_data = np.zeros((kNElectrodes, int(kNumPacketsToAcquire * kNSamplesPerPacket / kNElectrodes)), dtype=int)
    while True:
        packet = np.zeros(kNSamplesPerPacket, dtype=int)
        for i in range(0, kSizePacket, 2):
            low = q.get()
            high = q.get()
            sample = low + (high << 8)
            packet[int(i / 2)] = sample

        start_index = int(processed_packets / kNElectrodes) * kNSamplesPerPacket
        emg_data[electrode, start_index: start_index + kNSamplesPerPacket] = packet
        electrode = electrode + 1
        electrode = electrode % kNElectrodes
        processed_packets = processed_packets + 1
        if processed_packets >= kNumPacketsToAcquire:
            np.save(movement_class + "_" + str(kAcquisitionTime) + "s_" + str(kSamplePerSecond) + "Hz_" +
                    acquisition_number, emg_data)
            os._exit(0)


def main():
    port_open = False
    while not port_open:
        try:
            ser = serial.Serial("COM3", timeout=None, baudrate=115000, xonxoff=False, rtscts=False, dsrdtr=False)
            ser.flushInput()
            ser.flushOutput()
            port_open = True
        except:
            pass

    q = queue.Queue()
    consumer = threading.Thread(target=process_serial_buffer, args=(q,))
    consumer.start()
    try:
        while True:
            bytesToRead = ser.inWaiting()
            if bytesToRead != 0:
                data = ser.read(bytesToRead)
                for sample in data:
                    q.put(sample)
    except KeyboardInterrupt:
        print('interrupted!')
    os._exit(0)


if __name__ == "__main__":
    main()
