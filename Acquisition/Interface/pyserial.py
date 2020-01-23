
import serial
import time
import queue
import numpy as np

code = 1
pin_data=[]
valeurs = []
i = 0


arduino = serial.Serial('COM6', 115200, timeout=.1)
time.sleep(1) #give the connection a second to settle
arduino.write(code)

# q = queue.Queue()
while True:
	data = arduino.readline() #[:-2] #the last bit gets rid of the new-line chars
	print(data)
	# if data:
	# 	for sample in data :
	# 		print(sample)

	# i=0
	# for sample in data :
	# 	# pin_data.append(sample)
	# 	# pin_number = low
	# 	# print(pin_number)
	# 	print(sample)
	# 	q.put(sample)
	# 	# i = i+1
	# low = q.get()
	#
	# if(low == 0):
	# 	high = q.get()
	# 	if(high < 10) :
	# 		pin_number = high
	# 		print("pin number is " + str(pin_number))
	# else :
	# 	high = q.get()
	# 	value = low + (high << 8)
	# 	print("value = " + str(value))
	#
	# # valeurs.append(pin_data)
	# # print(valeurs)
