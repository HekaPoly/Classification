#Main class

import numpy as np
import LSTMAE
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from keras.utils import to_categorical

# fit and evaluate DNN or RF
def evaluate_model(X_train, y_train, X_test, y_test):

    print("TRAINING CLASSIFIER")

    #Neural network
    model = Sequential()
    model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(y_train.shape[1], activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=500, batch_size=10)

    _, accuracy = model.evaluate(X_test, y_test,verbose=1)
    print('Accuracy: %.2f' % (accuracy*100))

    y_pred = model.predict(X_test)

   
    y_pred = np.argmax(y_pred,axis = 1) 
    y_test = np.argmax(y_test,axis = 1)
    print(confusion_matrix(y_pred, y_test))
    print(classification_report(y_test, y_pred))

    model.save('models/DNN')



# main function to call
def run_experiment():
    file_name = "Dataset"    

    X_data, Y_data = LSTMAE.extract_features(file_name)

    Y_data = to_categorical(Y_data)

    X_train, X_test, y_train, y_test = train_test_split(
        X_data, Y_data, stratify=Y_data, test_size=0.30, random_state=42
    )

    evaluate_model(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    run_experiment()
   


