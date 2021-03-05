import numpy as np
import matplotlib.pyplot as plt
import keras
import os
import tensorflow as tf
from sklearn.preprocessing import normalize
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model, load_model
from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Dropout, Conv1D, Flatten, MaxPooling1D
from keras.utils import to_categorical
 
###################   PREPROCESSING SECTION  ##########################

########################################
config = tf.compat.v1.ConfigProto()    #
config.gpu_options.allow_growth = True #
tf.compat.v1.Session(config=config)    #
########################################

categories = [
    "HandOpen", "HandRest","ObjectGrip","PinchGrip","WristExten","WristFlex","WristPron", "WristSupi"
]

#Create 3d array for lstmae training
def create_time_series(filepath, n_timesteps):
    time_series = []
    Y = []
    first_y = True 
    first_windows = True
    for category in categories:
        print(category)
        data = np.load(filepath + "/" + category + ".npy")
        data = normalize(data, axis=1)
        windows = create_sliding_windows(data, n_timesteps)
        y_category = label_category(category, len(windows))
        if first_y:                                        
            Y = np.array(y_category)
            first_y = False
        else:
            Y = np.vstack((Y,y_category))

        if first_windows:                                   
            time_series = windows
            first_windows = False
        else:
            time_series = np.vstack((time_series, windows))

    return time_series, Y


# Create windows of n_timesteps BY ADDING PADDING AND THEN SPLITTING THE ARRAY
def split_into_windows(X, n_timesteps):
    windows = []

    # Fill array until it can be split into windows of n timesteps (elements)
    while len(X) % n_timesteps != 0:
        X = np.row_stack((X, X[ len(X) -1 ]))

    windows = np.array_split(X, (len(X) / n_timesteps), axis=0)
    return np.array(windows)


# Create sliding windows of n_timesteps (data augmentation technique)
def create_sliding_windows(data, n_timesteps):
    windows = []
    for i in range(data.shape[0] - n_timesteps):
      windows.append(data[i: i + n_timesteps])

    return np.array(windows)

#Label one category
def label_category(category, windows_length):
    indexLabel = categories.index(category)
    Y = np.full((windows_length, 1), indexLabel)
    return Y

#Visualize reconstruction of input data to evaluate the performance of the autoencoder
def visualize_reconstruction(X, model):
    X_predict = model.predict(X)
      
    fig, axs = plt.subplots(2)
    fig.suptitle('Reconstruction')
    for i in range(100, 150): # Choose intervals
      axs[0].plot(X[i])
      axs[1].plot(X_predict[i])

################### LSTMAE TRAINING SECTION  ##########################


# Extract features
def extract_features(filepath):
    n_timesteps = 30 #CHANGE TIMESTEPS HERE
    X, Y = create_time_series(filepath, n_timesteps)
    print("X.shape", X.shape)
    print("Y.shape", Y.shape)
    # X_encoded, model = evaluate_AE_model(X) #Evaluate model and extract features
    # visualize_reconstruction(X, model)
    # print("X_encoded.shape", X_encoded.shape)

    Y_categorical = to_categorical(Y) 
    # return X_encoded, Y_categorical
    return X, Y_categorical


# Create, train and evaluate LSTMAE model
def evaluate_AE_model(X):

    print("TRAINING LSTM AUTOENCODER")
    
    n_timesteps = X.shape[1] 
    n_features = X.shape[2]
   
    # Architecture utilisée lors de l'essai 5 (87% accuracy de classification)

    #create encoder part
    encoder_decoder = Sequential()
    encoder_decoder.add(LSTM(128, activation='relu', input_shape=(n_timesteps, n_features), return_sequences=True))
    encoder_decoder.add(LSTM(64, activation='relu', input_shape=(n_timesteps, n_features), return_sequences=True))
    encoder_decoder.add(LSTM(32, activation='relu', input_shape=(n_timesteps, n_features), return_sequences=False))

    #create decoder part
    encoder_decoder.add(RepeatVector(n_timesteps))
    encoder_decoder.add(LSTM(32, activation='relu', return_sequences=True))
    encoder_decoder.add(LSTM(64, activation='relu', return_sequences=True))
    encoder_decoder.add(LSTM(128, activation='relu', return_sequences=True))
    encoder_decoder.add(TimeDistributed(Dense(n_features)))
    opt = keras.optimizers.Adam(learning_rate=0.001)

    # Compile the model
    n_epochs = 10
    encoder_decoder.compile(optimizer=opt, loss='mse')
    encoder_decoder.summary()

    # Fit data to model
    encoder_decoder.fit(X, X, epochs=n_epochs,verbose=1, batch_size=100)

    #Save the model
    if os.name == 'nt': # Windows
        encoder_decoder.save(str(n_timesteps) + "timesteps_" + str(n_epochs) + "_sw" ) 
    else: # Linux/Mac
        encoder_decoder.save("../" + str(n_timesteps) + "timesteps_" + str(n_epochs) + "_sw" )
    
    #Extract features
    encoder = Model(inputs=encoder_decoder.inputs, outputs=encoder_decoder.layers[1].output)
    X_encoded = encoder.predict(X)

    return X_encoded, encoder_decoder
    pass

################### LSTM CLASSIFIER TRAINING SECTION  ##########################
# LR_START = 0.001
# LR_MAX = 0.060 #before 0.06
# LR_MIN = 0.0001
# LR_RAMPUP_EPOCHS = 65
# LR_SUSTAIN_EPOCHS = 2
# LR_EXP_DECAY = .9
LR_START = 0.001
LR_MAX = 0.0020
LR_MIN = 0.0005
LR_RAMPUP_EPOCHS = 45
LR_SUSTAIN_EPOCHS = 10
LR_EXP_DECAY = .9

def lrfn(epoch):
    if epoch < LR_RAMPUP_EPOCHS:
        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
        lr = LR_MAX
    else:
        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY**(epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN
    return lr

n_epochs = 125
lr = [lrfn(epoch) for epoch in range(n_epochs)]
plt.plot(lr)
plt.show()

def make_inception_model(n_timesteps, n_features, n_outputs):
    initializer = tf.random_normal_initializer(0.,0.2)
    input_layer = tf.keras.layers.Input(shape=(n_timesteps, n_features), name="inception-input")

    dropout = 0.45
    conv1 = Conv1D(64, 3, activation='relu')(input_layer)
    conv1 = Dropout(dropout)(conv1)
    conv2 = Conv1D(64, 4, activation='relu')(input_layer)
    conv2 = Dropout(dropout)(conv2)
    conv3 = Conv1D(64, 6, activation='relu')(input_layer)
    conv3 = Dropout(dropout)(conv3)
    x = keras.layers.Concatenate(axis=1)([conv1, conv2, conv3])
    
    conv1 = Conv1D(64, 3, activation='relu')(x)
    conv1 = Dropout(dropout)(conv1)
    conv2 = Conv1D(64, 4, activation='relu')(x)
    conv2 = Dropout(dropout)(conv2)
    conv3 = Conv1D(64, 6, activation='relu')(x)
    conv3 = Dropout(dropout)(conv3)
    max_pool = MaxPooling1D(pool_size=3)(x)
    x = keras.layers.Concatenate(axis=1)([conv1, conv2, conv3, max_pool])
    x = Flatten()(x)
    # x = Dropout(0.90)(x)
    x = Dense(100, activation='relu')(x)
    x = Dropout(dropout)(x)
    out = Dense(n_outputs, activation='sigmoid')(x)
    return Model(inputs=input_layer, outputs=out, name='inception-net1')




# fit and evaluate LSTM
def evaluate_model(X_train, y_train, X_test, y_test):

    print("TRAINING CLASSIFIER")
  
    n_timesteps = X_train.shape[1]
    n_features = X_train.shape[2]
    n_outputs = y_train.shape[1]

    # Architecture utilisée lors de l'essai 5 (87% accuracy)
    # model = Sequential()
    # model.add(LSTM(32, return_sequences=True,input_shape=(n_timesteps, n_features)))
    # model.add(Dropout(0.45))
    # model.add(LSTM(32, return_sequences=True))
    # model.add(Dropout(0.45))
    # model.add(LSTM(32, return_sequences=True))
    # model.add(Dropout(0.45))
    # model.add(LSTM(32))
    # model.add(Dense(128, activation="relu"))
    # model.add(Dropout(0.45))
    # model.add(Dense(n_outputs, activation="sigmoid"))

    # model = make_inception_model(n_timesteps, n_features, n_outputs)
    
    model = Sequential()
    model.add(Conv1D(512, 6, activation='relu', input_shape=(n_timesteps, n_features)))
    # model.add(Dropout(0.20))
    model.add(Conv1D(342, 4, activation='relu'))
    # model.add(Dropout(0.20))
    model.add(Conv1D(256, 3, activation='relu'))
    # model.add(Dropout(0.2))
    # model.add(LSTM(64))
    # model.add(Dropout(0.30))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    # model.add(Dropout(0.30))
    model.add(Dense(n_outputs, activation="sigmoid"))

    # Compile the model
    opt = keras.optimizers.Adam(learning_rate=0.001)
    n_epochs = 25
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'] )
    model.summary()

    # Fit data to model
    lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose = True)
    #model.fit(X_train, y_train, validation_data=(X_test, y_test),epochs=n_epochs, batch_size=500, callbacks=[lr_callback])
    model.fit(X_train, y_train, validation_data=(X_test, y_test),epochs=n_epochs, batch_size=500)
    _, accuracy = model.evaluate(X_test, y_test,verbose=1)
    print('Accuracy: %.2f' % (accuracy*100))


    #Save the model
    if os.name == 'nt': # Windows
        model.save(str(n_epochs) + "_" + str(accuracy * 100) + "_sw")
    else: #Linux/Mac
        model.save("../" + str(n_epochs) + "_" + str(accuracy * 100) + "_sw")

    # Generate generalization metrics
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred,axis = 1) 
    y_test = np.argmax(y_test,axis = 1)
    print(confusion_matrix(y_pred, y_test))
    print(classification_report(y_test, y_pred))
  
 ###################   MAIN SECTION  ##########################


# main function
def run_experiment():

    if os.name == 'nt': # Windows
        file_name = '..\\..\\Acquisition\\Data\\8mouvementOpenDataset'
    else: # Linux/Mac
        file_name = '../../Acquisition/Data/8mouvementOpenDataset'
    X_data, Y_data = extract_features(file_name)
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, Y_data, stratify=Y_data,test_size=0.20, random_state=42
    )
    evaluate_model(X_train, y_train, X_test, y_test)

run_experiment()

