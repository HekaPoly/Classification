import numpy as np
import tensorflow as tf
import keras

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Conv1D, Flatten

from sklearn.metrics import classification_report, confusion_matrix

########################################
config = tf.compat.v1.ConfigProto()    #
config.gpu_options.allow_growth = True #
tf.compat.v1.Session(config=config)    #
########################################

class ModelConv(object):
    def __init__(self, n_angles, n_electrods, n_timesteps):
        self.model = self._create_model(n_angles, n_electrods, n_timesteps)
        
    def _create_model(self, n_angles, n_electrods, n_timesteps):
        model = Sequential(name="conv_angle_v1")
        model.add(Conv1D(512, 6, activation='relu', input_shape=(n_timesteps, n_electrods)))
        model.add(Conv1D(342, 4, activation='relu'))
        model.add(Conv1D(256, 3, activation='relu'))
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(Dense(n_angles, activation='linear'))
        model.compile(loss='mse', optimizer='adam')
        return model

    def train(self, trainX, trainY, testX, testY, n_epochs, batch_size):
        self.model.fit(trainX, 
                       trainY, 
                       validation_data=(testX, testY), 
                       epochs=n_epochs, 
                       batch_size=batch_size)

    def predict(self, x):
        return self.model.predict(x)

    def evaluate(self, testX, testY):
        _, accuracy = self.model.evaluate(testX, testY, verbose=1)
        
        print('Accuracy: %.2f' % (accuracy*100))
        
        y_pred = self.predict(testX)
        y_pred = np.argmax(y_pred, axis=1)
        y_test = np.argmax(testY, axis=1)
        
        print(confusion_matrix(y_pred, y_test))
        print(classification_report(y_test, y_pred))


    def load(self, path):
        self.model = keras.models.load_model(path)

    def save(self, path):
        self.model.save(path)

if __name__=="__main__":
    model = ModelConv(1,2,60)
    model.model.summary()