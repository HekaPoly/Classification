#LSTM-Autoencoder Training

import numpy as np
import TimeSeries
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
from keras.models import load_model

#Big function to extract features
def extract_features(filepath):
    n_timesteps = 30 #CHANGE TIMESTEPS HERE
    X, Y = TimeSeries.create_time_series(filepath, n_timesteps)
    X_encoded = evaluate_AE_model(X) #Evaluate model and extract features
    return X_encoded, Y


# Create, train and evaluate model
def evaluate_AE_model(X):

    print("TRAINING LSTM AUTOENCODER")

    n_timesteps = X.shape[1] 
    n_features = X.shape[2]
   
    #create encoder part
    encoder_decoder = Sequential()
    encoder_decoder.add(LSTM(128, activation='relu', input_shape=(n_timesteps, n_features), return_sequences=True))
    encoder_decoder.add(LSTM(64, activation='relu', input_shape=(n_timesteps, n_features), return_sequences=False))
    # Le nombre de features extrait depend en fait du nombre de neurones du dernier layer de l encodeur 

    #create decoder part
    encoder_decoder.add(RepeatVector(n_timesteps))
    encoder_decoder.add(LSTM(64, activation='relu', return_sequences=True))
    encoder_decoder.add(LSTM(128, activation='relu', return_sequences=True))
    encoder_decoder.add(TimeDistributed(Dense(n_features)))

    encoder_decoder.compile(optimizer= "adam", loss='mse')
    encoder_decoder.fit(X, X, epochs=20,verbose=1)

    loaded_model = load_model("models/LSTMAE_30timesteps")

    #Extract features
    encoder = Model(inputs=loaded_model.inputs, outputs=loaded_model.layers[1].output)
    X_encoded = encoder.predict(X)

    return X_encoded