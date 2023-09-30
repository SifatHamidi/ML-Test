import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay
import tensorflow_addons as tfa
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D, BatchNormalization, \
    Conv2D, MaxPooling2D, ZeroPadding2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import load_model


# Load Mnist dataset
mnist_data = tf.keras.datasets.mnist.load_data()


# Split the data into train and test
(x_train, y_train), (x_test, y_test) = mnist_data
print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)



# Visualize some mnist data sample
index_values = [25, 39, 8, 67, 90, 5120, 1000, 234, 1267, 132]
plt.figure(figsize=(6, 6))
for i in range(len(index_values)):
    plt.subplot(3, 4, i + 1)
    plt.imshow(x_train[a[i]])
    plt.title(y_train[a[i]])



# Reshaping the data
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)


# Conver the classes into categorical values.The categorial output contains contains 10 class
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)



# Convolutional Neural Network architecture to train the model.The input image size is 28x28.

model = Sequential()
# 1st hidden layer
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
# 2nd hidden layer
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))
# Fulley Conneacted Layers
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))



# Metrics variabale contains some model evaluation attributes
metrics = ['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall(),
           tfa.metrics.CohenKappa(num_classes=10), tfa.metrics.F1Score(num_classes=10)]

# 'categorical_crossentropy'- for multiclass classification
model.compile(optimizer=Adam(lr=3e-4), loss='categorical_crossentropy',
              metrics=metrics)
model.summary()



# Some Model regularization technique to check overfitting,tracking model weights and training
early_stopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, restore_best_weights=True)

checkpointer = ModelCheckpoint(filepath='/kaggle/working/model.hdf5',
                               monitor='val_loss',
                               save_best_only=True,
                               mode='auto', verbose=1)

reducelr = ReduceLROnPlateau(monitor='val_loss',
                             factor=0.2,
                             patience=5,
                             min_lr=0.0002)



# Fit the model with above mentioned techniques
history = model.fit(x_train, y_train, batch_size=60, epochs=25, validation_split=0.2,
                    callbacks=[checkpointer, early_stopper, reducelr])

pd.DataFrame(history.history).plot()
plt.show()

# Trained Model Evaluation
result = model.evaluate(x_test, y_test)
print('Accuracy:', result[1])
print('Loss:', result[0])
print('Precision:', result[3])
print('Recall:', result[4])


y_pred_classes = np.argmax(model.predict(x_test), axis=1)
y_true = np.argmax(y_test, axis=1)
report = classification_report(y_true, y_pred_classes)
print('classification report', report)



labels = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}
confusion_matrix_result = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(8, 8))
sns.heatmap(confusion_matrix_result, annot=True, fmt="d", cmap="Blues", xticklabels=labels.keys(),
            yticklabels=labels.keys())
plt.show()



# Evaluate the model on unknown test data

models = load_model('/kaggle/working/model.hdf5')
num = 115
prediction = models.predict(x_test)
print(np.argmax(np.round(prediction[num])))

plt.imshow(x_test[num].reshape(28, 28))
print("The Label is '{}'".format(y_test[num]))
plt.show()
