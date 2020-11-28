from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
import classes
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from copy import deepcopy


x_data = np.load("features_demo2.npy")
y_data = np.load("y_demo.npy")

# kNExecutions = 10
# avg = 0
# best_score = 0
# best_ANN = ANN = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(40, 100))
# for i in range(kNExecutions):
X_train, X_test, y_train, y_test = train_test_split(x_data,y_data, test_size = 0.01, random_state = 32)
ANN = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(40, 100))
# np.repeat(X_train, 10000, axis=0)
# np.repeat(y_train, 10000, axis=0)
ANN.fit(X_train, y_train)

ANN_y_predict = ANN.predict(X_test)

ANN_score = 0
for idx, result in enumerate(ANN_y_predict):
    try:
        one_indexes = np.where(ANN_y_predict[idx] == 1)
        if len(one_indexes[0]) > 1:
            for i in range(len(one_indexes[0]) - 1):
                ANN_y_predict[idx, one_indexes[0][i]] = 0

        if (np.where(ANN_y_predict[idx] == 1) == np.where(y_test[idx] == 1)):
            ANN_score += 1
    except:
        pass

    # ANN_score = ANN_score / ANN_y_predict.shape[0]
    # avg += ANN_score
    # print(ANN_score)
    # if ANN_score > best_score:
    #     best_score = ANN_score
    #     best_ANN = deepcopy(ANN)


demo_predict = ANN.predict(np.load("features_demo2.npy"))
for idx, result in enumerate(demo_predict):
    try:
        one_indexes = np.where(demo_predict[idx] == 1)
        if len(one_indexes[0]) > 1:
            for i in range(len(one_indexes[0]) - 1):
                demo_predict[idx, one_indexes[0][i]] = 0
    except:
        pass
np.save("demo_predictions2", demo_predict)


#
# print("100 executions")
# print("avg = ", avg/kNExecutions)
# print("best = ", best_score)
#print(confusion_matrix(y_test.argmax(axis=1), ANN_y_predict.argmax(axis=1)))



# SVM = svm.SVC(gamma='scale')
# SVM.fit(X_train, y_train)
#
# SVM_y_predict = SVM.predict(X_test)
#
# SVM_score = 0
# for idx, result in enumerate(SVM_y_predict):
#     try:
#         if np.where(SVM_y_predict[idx] == 1) == np.where(y_test[idx] == 1):
#             SVM_score += 1
#     except:
#         pass
# print(SVM_score / SVM_y_predict.shape[0])
#
# print(confusion_matrix(y_test.argmax(axis=1), SVM_y_predict.argmax(axis=1)))

#
# RF = RandomForestClassifier(n_estimators=100, max_depth=2,
#                              random_state=0)
# RF.fit(X_train, y_train)
#
# RF_y_predict = RF.predict(X_test)
#
# RF_score = 0
# for idx, result in enumerate(RF_y_predict):
#     try:
#         if np.where(RF_y_predict[idx] == 1) == np.where(y_test[idx] == 1):
#             RF_score += 1
#     except:
#         pass
# print(RF_score / RF_y_predict.shape[0])
#
# print(confusion_matrix(y_test.argmax(axis=1), RF_y_predict.argmax(axis=1)))
