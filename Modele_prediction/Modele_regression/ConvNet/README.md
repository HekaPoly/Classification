# Convnet

[comment]: <> (TODO: faire quelque chose de plus cute)

We propose a convnet deeplearning network using Electromyography(EMG) data as input to predict the angle of different motors. These motors will be the main driving unit for an exoskeleton to help people suffering from muscular dystrophy.  

## Datasets
### Open dataset

#### Test data
#### Validation data
#### Training summary
```
rain on 470055 samples, validate on 56135 samples
Epoch 1/11
470055/470055 [==============================] - 145s 309us/sample - loss: 377.2329 - mae: 14.0596 - val_loss: 306.3630 - val_mae: 13.0634
Epoch 2/11
470055/470055 [==============================] - 137s 290us/sample - loss: 313.5652 - mae: 12.9307 - val_loss: 300.1451 - val_mae: 12.9393
Epoch 3/11
470055/470055 [==============================] - 137s 292us/sample - loss: 307.1037 - mae: 12.7902 - val_loss: 295.6211 - val_mae: 12.8325
Epoch 4/11
470055/470055 [==============================] - 137s 292us/sample - loss: 302.6736 - mae: 12.6908 - val_loss: 294.8547 - val_mae: 12.7836
Epoch 5/11
470055/470055 [==============================] - 1750s 4ms/sample - loss: 299.3118 - mae: 12.6119 - val_loss: 297.2305 - val_mae: 12.8611
Epoch 6/11
470055/470055 [==============================] - 136s 289us/sample - loss: 296.0841 - mae: 12.5343 - val_loss: 297.1645 - val_mae: 12.8309
Epoch 7/11
470055/470055 [==============================] - 135s 286us/sample - loss: 293.3994 - mae: 12.4684 - val_loss: 303.9916 - val_mae: 13.0333
Epoch 8/11
470055/470055 [==============================] - 33326s 71ms/sample - loss: 290.9151 - mae: 12.4077 - val_loss: 301.9860 - val_mae: 12.9014
Epoch 9/11
470055/470055 [==============================] - 149s 316us/sample - loss: 288.7413 - mae: 12.3553 - val_loss: 299.4428 - val_mae: 12.8910
Epoch 10/11
470055/470055 [==============================] - 138s 293us/sample - loss: 286.5063 - mae: 12.3016 - val_loss: 298.6201 - val_mae: 12.8457
Epoch 11/11
470055/470055 [==============================] - 132s 282us/sample - loss: 284.5374 - mae: 12.2536 - val_loss: 297.1339 - val_mae: 12.7899
56135/1
Accuracy: 1278.99
[[    0     0     0     0     0     0     0     0     0     0     0     0
      0     0     0     0]
 [    0     0     0     0     0     0     0     0     0     0     0     0
      0     0     0     0]
 [    0     0     0     0     0     0     0     0     0     0     0     0
      0     0     0     0]
 [    0     0     0     0     0     0     0     0     0     0     0     0
      0     0     0     0]
 [  159     0     0     0   490   100    52     3     0   398   472     0
     47   182     5     0]
 [    0     0     0     0     0    18     0     0     0     0     0     0
      0     1     0     0]
 [  101     1     0    25    54    64    44     0     0   110   117     5
     20    17    22     0]
 [    0     0     0     0     0     0     0     0     0     0     0     0
      0     0     0     0]
 [    0     0     0     0     0     4     0     0     0     0     0     0
      0     0     0     0]
 [  279    89     9   308  8286  6868  3229    71   294 17955  5639    12
    973  7609   699    49]
 [   72     0     0     0   177    13    13     0     0   127   446    16
     28   104     0     0]
 [    0     0     0     0     0     0     0     0     0     0     0     0
      0     0     0     0]
 [    0     0     0     0     0     0     0     0     0     0     0     0
      0     0     0     0]
 [   12     0     0     0    40     0     0     0     0    30     1     0
      0    17     0     0]
 [   30     0     0     0    14     0     0     0     0     0     6    68
      0     3    38     0]
 [    0     0     0     0     0     0     0     0     0     0     0     0
      0     0     0     0]]
      
              precision    recall  f1-score   support

           1       0.00      0.00      0.00       653
           3       0.00      0.00      0.00        90
           4       0.00      0.00      0.00         9
           5       0.00      0.00      0.00       333
           6       0.26      0.05      0.09      9061
           7       0.95      0.00      0.01      7067
           8       0.08      0.01      0.02      3338
           9       0.00      0.00      0.00        74
          10       0.00      0.00      0.00       294
          11       0.34      0.96      0.51     18620
          12       0.45      0.07      0.12      6681
          13       0.00      0.00      0.00       101
          14       0.00      0.00      0.00      1068
          15       0.17      0.00      0.00      7933
          16       0.24      0.05      0.08       764
          17       0.00      0.00      0.00        49

    accuracy                           0.34     56135
   macro avg       0.15      0.07      0.05     56135
weighted avg       0.36      0.34      0.20     56135
```
### Metis dataset
#### Test data
#### Validation data


## Model architecture
### Why a convnet
### Layers
### Result

## Tutorial
### How to train the model
#### With the open dataset
[comment]: <> ( TODO: change the notebook to let the user choose a file)
Simply run the jupyter notebook named `train_open_dataset.ipynb`. Make sure to have the [dataset](https://github.com/MetisPoly/Classification/tree/develop/Acquisition/Data/Dataset_avec_angles_tester) at the correct path in your local project. 

### Tutorial: How to infer a motor angle


## Currently known issue with the proposed model
### Does model overfit?
### note pendant la rencontre avec cathia
- Model is too global
- Work well with central angle but not angle with big variation
  - Voir loss function pour plus prendre en compte les erreurs
  - voir batch normalization (si donnee trop normaliser, prend moins en compte les donnees extremes) (batch normalization local)
  
## File structure
- **conv_angle_v1_sw**
  - **variables**
    - **variables.data-00000-of-00001**
    - **variables.index**
  - **saved_model.pb:** convnet model using the [pb extension](https://www.tensorflow.org/guide/saved_model). This can be exported to be used in the embedded system.
- **src**
  - **.ipynb_checkpoints**
    - **convnet-checkpoint.py**
    - **Untitled-checkpoint.ipynb**
  - **Lib**
  - **convnet.py**
  - **dataset_conversion.py**
  - **GraphGenerator.py**
  - **train.py**: Train the model with an [open dataset](https://github.com/MetisPoly/Classification/tree/develop/Acquisition/Data/Dataset_avec_angles_tester)
  - **train_Metis.py:** Train the model with the [metis dataset](https://github.com/MetisPoly/Classification/tree/develop/Acquisition/Data/7_electrodes_Philippe/regroupement_des%20donnes_par_categorie/test_data) 
  - **Untitled.ipynb**
- **README.md**
