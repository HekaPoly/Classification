
import serial
import time

code = 1

arduino = serial.Serial('COM6', 115200, timeout=.1)
time.sleep(1) #give the connection a second to settle
arduino.write(code)

while True:
	data = arduino.readline() [:-2] #the last bit gets rid of the new-line chars
	if data:
		print(data)
