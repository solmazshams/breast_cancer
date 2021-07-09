import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse.construct import random
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras.metrics import categorical_accuracy, binary_accuracy
from sklearn.datasets import load_breast_cancer
from keras.callbacks import EarlyStopping

breast_cancer = load_breast_cancer()

features = breast_cancer["data"]
target = breast_cancer["target"]
print(features.shape)
df = pd.DataFrame(features, columns=breast_cancer.feature_names)
df["target"] = target

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.1, random_state = 123)

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(features.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, validation_split = 0.3, batch_size=5, epochs=200)
y_pred = model.predict(X_test)

print(model.summary)

# plt.hist(y_train, label='train')
# plt.hist(y_test, label='test')
# plt.hist(y_pred, label='pred')
# plt.legend()
# plt.show()

print("maximum accuracy : {}".format(np.max(history.history['accuracy'])))
print("maximum value accuracy : {}".format(np.max(history.history['val_accuracy'])))
print("minimum_loss : {}".format(np.min(history.history['loss'])))

plt.figure()
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.plot(history.history["loss"])
plt.title("accuracy and loss trends")
plt.xlabel("epoch")
plt.ylabel("accuracy-vs-loss")
plt.legend(['train_acc', 'test_acc', 'loss'])
plt.savefig('breast_cancer_deep_learning_accuracy.png')