import numpy as np
import audiomentations
import matplotlib.pyplot as plt
import random
import os
import math


def create_sin_mask(file, frequency, nb_spasm, spasm_lenght, amplitude):
    mask = np.ones_like(file)
    spasm_nb_point = math.ceil(frequency * spasm_lenght)

    for layer in range(file.shape[1]):
        chosen_position = []
        for i in range(nb_spasm):
            # essaye de le place sans causer d<overlap
            for _ in range(5):
                decallage = random.randint(0, file.shape[0])
                is_overlapping = False

                for element in chosen_position:
                    if decallage - spasm_nb_point < element < decallage + spasm_nb_point:
                        is_overlapping = True

                if not is_overlapping:
                    chosen_position.append(decallage)

                    for j in range(spasm_nb_point):
                        if (decallage + j) < file.shape[0]:
                            mask[decallage + j, layer] = amplitude * math.sin(j / spasm_nb_point * math.pi) + 1
                    break

    return mask


def modify_file(file, frequency, nb_spasm, spasm_lenght, amplitude):
    mask = create_sin_mask(file, frequency, nb_spasm, spasm_lenght, amplitude)

    return file * mask


if __name__ == '__main__':
    folder_path = "./data"
    show_plot = False
    save_file = True

    files = os.listdir(folder_path)

    for file_name in files:
        file = np.load(folder_path + "/" + file_name)
        modified_file = modify_file(file, 2000, 25, 2, 3)

        if show_plot:
            fig, axes = plt.subplots(2, 1)
            axes[0].plot(file)
            axes[1].plot(modified_file)
            plt.show()

        if save_file:
            np.save("./modifiedData/" + file_name, modified_file)