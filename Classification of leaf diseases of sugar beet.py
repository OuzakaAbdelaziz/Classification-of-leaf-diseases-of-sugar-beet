# Importation des librairies
import tensorflow as tf
import pandas as pd
import numpy as np
import sklearn
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.models import Model
from sklearn import metrics 
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib
from matplotlib import pyplot as plt

# Importation des données:
data= pd.read_csv('C:/Users/toshiba/Desktop/STE_S3/Agrogeomatique/TP2_Phytosanitaire/data.csv', sep=';')
data.head()

# Transformation des données en format matricielle :
data=data.to_numpy()
print('le nombre de ligne de dataset est:',data.shape[0])
print('le nombre de colonne de dataset est:',data.shape[1])

# Séparation des données en données d'entrainement et de test
train_set=data[(0,1,2,3,4,5,6,7,10,11,12,13,14,15,16,17,20,21,22,23,24,25,26,27),:]
test_set=data[(8,9,18,19,28,29),:]
print('train set',train_set.shape)
print('test set',test_set.shape)

# Séparation entre les features et labels
train_features= train_set[:,:-1]
train_label=train_set[:,-1]
test_features= test_set[:,:-1]
test_label=test_set[:,-1]

#print(train_features.shape)
#print(test_features.shape)
#print(train_label.shape)


# Création du model deep learning
mon_ANN = Sequential()
# Ajout de couches de type dense
mon_ANN.add(Dense(12, input_dim=255, activation='relu'))
mon_ANN.add(Dense(8, activation='relu'))
mon_ANN.add(Dense(8, activation='relu'))
mon_ANN.add(Dense(1, activation='sigmoid'))

# Compilation du model
optimiseur = tf.keras.optimizers.RMSprop(learning_rate = 0.001)
mon_ANN.compile(loss='binary_crossentropy', optimizer=optimiseur, metrics=['mae'])
early_stop = EarlyStopping(monitor='val_mae',patience=2)

# Entrainement sur les données train 
mon_ANN.fit(train_features,train_label ,validation_data=(test_features,test_label), epochs=100, batch_size=6,callbacks=[early_stop])

# Impression d'une description du réseau

plt.plot(mon_ANN.history.history['mae'])
plt.plot(mon_ANN.history.history['val_mae'])
plt.title('Entrainement et validation du model')
plt.ylabel('Erreur absolue moyenne')
plt.xlabel('Epochs')
plt.legend(['Train','validation'], loc='upper right')


# Metrics de validation
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(train_features,train_label)
predictions = model.predict(test_features)
print('les classes de test sont : ',test_label)
print('les classes prédictées sont :',predictions)
print(classification_report(test_label, predictions))
conf_matrix=confusion_matrix(test_label, predictions)


# Tracer la matrice de confusion :
fig, ax = plt.subplots(figsize=(6, 6))
ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predicted', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()