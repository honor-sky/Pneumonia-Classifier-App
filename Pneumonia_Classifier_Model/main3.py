import random, math
import cv2
import os
import numpy as np
import pandas as pd
from tensorflow import optimizers
import tensorflow as tf

import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization, GlobalMaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report,confusion_matrix
from keras.callbacks import ReduceLROnPlateau


# GPU 사용하여 의미 없음
seed = 100
np.random.seed(seed)

img_size = 150
labels = ['PNEUMONIA', 'NORMAL']
METRICS = [
    'accuracy',
    tf.keras.metrics.Recall(name='recall')
]

'''
def get_training_data(data_dir):
    data = []
    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return np.array(data)

# load image
train = get_training_data('C:/Users/DKU/PycharmProjects/Final_project/data/chest_xray/train')
test = get_training_data('C:/Users/DKU/PycharmProjects/Final_project/data/chest_xray/test')
val = get_training_data('C:/Users/DKU/PycharmProjects/Final_project/data/chest_xray/val')


# validation dataset 추가 (trian 데이터의 20%를 더해줌)
train_size = int(len(train)*0.8)
train_index = random.sample(range(0, len(train)),train_size)
val_index=[]

for i in range(0,len(train)):
    if(i in train_index):
        continue
    else:
        val_index.append(i)

val = train[val_index]
train = train[train_index]


# 데이터셋 저장(데이터를 불러오는 시간 감축)

np.save('C:/Users/DKU/PycharmProjects/Final_project/train', train)
np.save('C:/Users/DKU/PycharmProjects/Final_project/test', test)
np.save('C:/Users/DKU/PycharmProjects/Final_project/val', val)

'''
# 미리 만들어둔 train, test, val 데이터셋 불러옴
train = np.load('C:/Users/DKU/PycharmProjects/Final_project/train.npy',allow_pickle=True)
test = np.load('C:/Users/DKU/PycharmProjects/Final_project/test.npy',allow_pickle=True)
val = np.load('C:/Users/DKU/PycharmProjects/Final_project/val.npy',allow_pickle=True)


# split data to feature/label
x_train = []
y_train = []

x_val = []
y_val = []

x_test = []
y_test = []

for feature, label in train:
    x_train.append(feature)
    y_train.append(label)

for feature, label in test:
    x_test.append(feature)
    y_test.append(label)

for feature, label in val:
    x_val.append(feature)
    y_val.append(label)

# Normalize the data
x_train = np.array(x_train) / 255
x_val = np.array(x_val) / 255
x_test = np.array(x_test) / 255

# resize data for deep learning
x_train = x_train.reshape(-1, img_size, img_size, 1)
y_train = np.array(y_train)

x_val = x_val.reshape(-1, img_size, img_size, 1)
y_val = np.array(y_val)

x_test = x_test.reshape(-1, img_size, img_size, 1)
y_test = np.array(y_test)



# ImageDataGenerator 활용
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whiteninㅋ
        zoom_range = 0.2, # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        vertical_flip=False)  # randomly flip images
datagen.fit(x_train)

# create model
model = Sequential()
model.add(Conv2D(32 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (150,150,1)))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Conv2D(64 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(Dropout(0.1))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Conv2D(128 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Conv2D(256 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Conv2D(512 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(Dropout(0.1))
model.add(BatchNormalization())
model.add(GlobalMaxPooling2D())
model.add(Dense(units = 256 , activation = 'relu'))
model.add(Dense(units = 128 , activation = 'relu'))
model.add(Dropout(0.1))
model.add(Dense(units = 1 , activation = 'sigmoid'))  #sigmoid
model.compile(optimizer = optimizers.SGD(learning_rate=1e-3,momentum=0.9) , loss = 'binary_crossentropy' , metrics = ['accuracy'])  #"rmsprop"  # learning_rate=1e-4
model.summary()

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 2, verbose=1,factor=0.3, min_lr=0.000001)

# train model
history = model.fit(datagen.flow(x_train,y_train,batch_size=32) ,epochs = 50 , validation_data = datagen.flow(x_val, y_val) ,callbacks = [learning_rate_reduction])

# print testing result
print("Loss of the model is - " , model.evaluate(x_test,y_test)[0])
print("Accuracy of the model is - " , model.evaluate(x_test,y_test)[1]*100 , "%")


# save model
h5_path = 'C:/Users/DKU/PycharmProjects/Final_project/model/myModel6.h5'
pb_path = 'C:/Users/DKU/PycharmProjects/Final_project/model/myModel6.pb'
tflite_path = 'C:/Users/DKU/PycharmProjects/Final_project/model/myModel6.tflite'


model.save(h5_path)  # 안드로이드 내장을 위해 h5 -> pb -> tflite 형태로 바꿔줘야 함

h5_model = tf.keras.models.load_model(h5_path)
h5_model.save(pb_path, save_format="tf")

converter = tf.lite.TFLiteConverter.from_saved_model(pb_path)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()
open(tflite_path, 'wb').write(tflite_model)



# draw training graph
epochs = [i for i in range(50)]
fig , ax = plt.subplots(1,2)
train_acc = history.history['accuracy']
train_loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']
fig.set_size_inches(20,10)

ax[0].plot(epochs , train_acc , 'go-' , label = 'Training Accuracy')
ax[0].plot(epochs , val_acc , 'ro-' , label = 'Validation Accuracy')
ax[0].set_title('Training & Validation Accuracy')
ax[0].legend()
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Accuracy")

ax[1].plot(epochs , train_loss , 'g-o' , label = 'Training Loss')
ax[1].plot(epochs , val_loss , 'r-o' , label = 'Validation Loss')
ax[1].set_title('Training & Validation Loss')
ax[1].legend()
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Training & Validation Loss")
plt.show()

'''
#predictions = (model.predict(x_test) > 0.35).astype("int32")
#predictions = np.argmax(model.predict(x_test),axis=1) # axis=-1
predictions = model.predict(x_test)
predictions = labels[np.argmax()]
print(predictions)
#predictions = np.round(predictions).astype(int)
predictions = predictions.reshape(1,-1)[0]
predictions[:15]
print(predictions)

print(classification_report(y_test, predictions, target_names = ['Pneumonia (Class 0)','Normal (Class 1)']))

cm = confusion_matrix(y_test,predictions)
cm
cm = pd.DataFrame(cm , index = ['0','1'] , columns = ['0','1'])

plt.figure(figsize = (10,10))
sns.heatmap(cm,cmap= "Blues", linecolor = 'black' , linewidth = 1 , annot = True, fmt='',xticklabels = labels,yticklabels = labels)
plt.show()

correct = np.nonzero(predictions == y_test)[0]
incorrect = np.nonzero(predictions != y_test)[0]
print(correct)
print(incorrect)
'''