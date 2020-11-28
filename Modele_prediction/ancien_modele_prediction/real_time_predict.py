import numpy as np
import time
import classes
from tkinter import *
from threading import Thread


root = Tk()
# MainWindow(root)
canvas = Canvas(width=300, height=200, bg='white')
canvas.pack(expand=YES, fill=BOTH)
images = {}
for class_name in classes.names:
    images[class_name] = PhotoImage(file="../img/"+class_name+".png")
canimage = canvas.create_image(0, 0, anchor=NW, image=images["shoulder_abduction_"])


def iterate_predictions():
    kWindowSize = 0.1
    demo_predict = np.load("demo_predictions2.npy")
    for index, line in enumerate(demo_predict):
        time.sleep(kWindowSize)
        print(line)
        if len(np.where(demo_predict[index] == 1)[0]) >= 1:
            classname = classes.to_string(np.where(demo_predict[index] == 1)[0][0])
            # if "rest" not in classname:
            canvas.itemconfig(canimage, image=images[classname])
    root.quit()


# root.after(1, iterate_predictions)
thread = Thread(target=iterate_predictions)
thread.start()
root.mainloop()
