from slidingWindow import SlidingWindow
from sklearn.model_selection import train_test_split
from model import Model
import os

if __name__ == '__main__':
    # relative file path to the training data
    file_path = "data"

    # file path to the saved model
    saved_model_path = "model"


    categories = ["HandOpen", "HandRest", "ObjectGrip", "PichGrip", "WristExten", "WristFlex", "WristPron", "WristSupi"]

    sliding_window = SlidingWindow(categories)

    # Format the data
    X, Y = sliding_window.extract_features(file_path, 15)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=27)

    # parameter for the model
    n_layers = 3
    d_model = X.shape[2]
    n_heads = 7
    units = 1024
    dropout = 0.1
    time_steps = X.shape[1]
    output_size = 8
    epochs = 20

    model = Model(time_steps, n_layers, units, d_model, n_heads, dropout, output_size)

    model.train_model(x_train, y_train, x_test, y_test, epochs, saved_model_path)


