import matplotlib.pyplot as plt
import numpy as np
import cv2


class_names = ['CUP', 'SPOON', 'FORK', 'MOUSE']

#creating realtime dataset

CAMERA = cv2.VideoCapture(0)
camera_height = 500

raw_frames_type_1 = [ ]
raw_frames_type_2 = [ ]
raw_frames_type_3 = [ ]
raw_frames_type_4 = [ ]

while CAMERA.isOpened():
    #read a new camera frame

    ret, frame = CAMERA.read()

    #Flip

    frame = cv2.flip(frame, 1)

    #Rescale the images output

    aspect = frame.shape[1]/float (frame.shape[0])
    res = int(aspect * camera_height)
    frame = cv2.resize(frame, (res, camera_height))

    #the green rectangle frame
    cv2.rectangle(frame, (300, 75), (650, 425), (0, 255, 0), 2)

    cv2.imshow ("Capturing", frame)

    #controls q = quit / s = capturing

    key = cv2.waitKey(1)

    if key & 0xff == ord ('q'):
        break
    elif key & 0xff == ord ('1'):
        #save the raw frames to frame
        raw_frames_type_1.append(frame)
    elif key & 0xff == ord ('2'):
        #save the raw frames to frame
        raw_frames_type_2.append(frame)
    elif key & 0xff == ord ('3'):
        #save the raw frames to frame
        raw_frames_type_3.append(frame)
    elif key & 0xff == ord ('4'):
        #save the raw frames to frame
        raw_frames_type_4.append(frame)

    plt.imshow(frame)
    plt.show()

#camera
CAMERA.release()
cv2.destroyAllWindows



save_width = 339
save_height = 400

from glob import glob
import os

retval = os.getcwd()
print ("Currently Working Directory %s" % retval)



print ('img1:', len(raw_frames_type_1))
print ('img2:', len(raw_frames_type_2))
print ('img3:', len(raw_frames_type_3))
print ('img4:', len(raw_frames_type_4))

#crop the images

#for frames type 1
for i, frame in enumerate (raw_frames_type_1):
    #get roi
    roi = frame[75+2:425-2, 300+2:650-2]
    #parse BRG to RGB
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    #resize to 224 x 224
    roi = cv2.resize(roi, (save_width, save_height))
    #save 
    cv2.imwrite ('img_1/{}.png'.format(i), cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))

#for frames type 2
for i, frame in enumerate (raw_frames_type_2):
    #get roi
    roi = frame[75+2:425-2, 300+2:650-2]
    #parse BRG to RGB
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    #resize to 224 x 224
    roi = cv2.resize(roi, (save_width, save_height))
    #save 
    cv2.imwrite ('img_2/{}.png'.format(i), cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))

#for frames type 3
for i, frame in enumerate (raw_frames_type_3):
    #get roi
    roi = frame[75+2:425-2, 300+2:650-2]
    #parse BRG to RGB
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    #resize to 224 x 224
    roi = cv2.resize(roi, (save_width, save_height))
    #save 
    cv2.imwrite ('img_3/{}.png'.format(i), cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))

#for frames type 4
for i, frame in enumerate (raw_frames_type_4):
    #get roi
    roi = frame[75+2:425-2, 300+2:650-2]
    #parse BRG to RGB
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    #resize to 224 x 224
    roi = cv2.resize(roi, (save_width, save_height))
    #save 
    cv2.imwrite ('img_4/{}.png'.format(i), cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))



from glob import glob
from keras import preprocessing

width = 96
height = 96

images_type_1 = [ ]
images_type_2 = [ ]
images_type_3 = [ ]
images_type_4 = [ ]

for image_path in glob('img_1/*.*'):
    image = preprocessing.image.load_img(image_path, target_size = (width, height))
    x = preprocessing.image.img_to_array(image)

    images_type_1.append(x)
    
for image_path in glob('img_2/*.*'):
    image = preprocessing.image.load_img(image_path, target_size = (width, height))
    x = preprocessing.image.img_to_array(image)

    images_type_2.append(x)
    
for image_path in glob('img_3/*.*'):
    image = preprocessing.image.load_img(image_path, target_size = (width, height))
    x = preprocessing.image.img_to_array(image)

    images_type_3.append(3)
    
for image_path in glob('img_4/*.*'):
    image = preprocessing.image.load_img(image_path, target_size = (width, height))
    x = preprocessing.image.img_to_array(image)

    images_type_4.append(x)

plt.figure(figsize=(12, 8))




for i, x in enumerate(images_type_1[:5]):
    plt.subplot(1, 5, i+1)
    image = preprocessing.image.array_to_img(x)
    plt.imshow(image)

    plt.axis('off')
    plt.title('{} image'.format(class_names[0]))

plt.show()

for i, x in enumerate(images_type_2[:5]):
    plt.subplot(1, 5, i+1)
    image = preprocessing.image.array_to_img(x)
    plt.imshow(image)

    plt.axis('off')
    plt.title('{} image'.format(class_names[0]))

plt.show()

for i, x in enumerate(images_type_3[:5]):
    plt.subplot(1, 5, i+1)
    image = preprocessing.image.array_to_img(x)
    plt.imshow(image)

    plt.axis('off')
    plt.title('{} image'.format(class_names[0]))

plt.show()

for i, x in enumerate(images_type_4[:5]):
    plt.subplot(1, 5, i+1)
    image = preprocessing.image.array_to_img(x)
    plt.imshow(image)

    plt.axis('off')
    plt.title('{} image'.format(class_names[0]))

plt.show()












#prepare image to tensor


X_type_1 = np.array(images_type_1)
X_type_2 = np.array(images_type_2)
X_type_3 = np.array(images_type_3)
X_type_4 = np.array(images_type_4)

#check the shape using .shape() check the images count

print (X_type_1.shape)
print (X_type_2.shape)
print (X_type_3.shape)
print (X_type_4.shape)


(13, 96, 96, 3)
(23, 96, 96, 3)
(14, 96, 96, 3)
(22, 96, 96, 3)

X_type_2


X = np.concatenate(X_type_1, X_type_2, axis= 0)

if len(X_type_3):
    X = np.concatenate((X, X_type_3), axis= 0)
if len (X_type_4):
    X = np.concatenate((X, X_type_3), axis= 0)

    X = X/250

    X.shape



from keras.utils import to_categorical

y_type_1 = [0 for item in enumerate(X_type_1)]
y_type_2 = [0 for item in enumerate(X_type_2)]
y_type_3 = [0 for item in enumerate(X_type_3)]
y_type_4 = [0 for item in enumerate(X_type_4)]

y = np.concatenate((y_type_1, y_type_2), axis= 0)

if len(y_type_3):
    y = np.concatenate((y, y_type_3), axis= 0)
if len(y_type_4):
    y = np.concatenate((y, y_type_4), axis= 0)

y = to_categorical (y, num_classes=len(class_names))


y.shape







#CNN config

from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Flatten, Dense
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import Adam


#default parameters

#situational - values, you may not adjust here

conv_1 = 16
conv_1_drop = 0.2
conv_2 = 32
conv_2_drop = 0.2
dense_1_n = 1024
dense_1_drop = 0.2
dense_2_n = 512
dense_2_drop = 0.2

#values you can adjust
lr = 0.001
epochs = 5
batch_size = 10
color_channels = 3


def build_model(conv_1_drop = conv_1_drop, conv_2_drop = conv_2_drop,
                dense_1_n = dense_1_n, dense_1_drop = dense_1_drop,
                dense_2_n = dense_2_n, dense_2_drop = dense_2_drop, lr = lr):
    
    model = Sequential()

    model.add(Convolution2D(conv_1, (3,3),
                            input_shape = (width, height, color_channels),
                            activation='relu'))
    
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(conv_1_drop))
    
    #---
    model.add(Flatten())

    #---
    model.add(Dense(dense_1_n, activation='relu'))
    model.add(Dropout(dense_1_drop))

    #---
    model.add(Dense(dense_2_n, activation= 'relu'))
    model.add(Dropout(dense_2_drop))

    #---
    model.add(Dense(len(class_names), activation= 'relu'))

    model.compile(loss = 'categorical_crossentropy',
                optimizer= Adam(clipvalue = 0.5),
                metrics=['accuracy'])
    
    return model

    #model parameter

    model = build_model()
    model.summary()


    #do not run yet

    history = model.fit(X, y, validation_split = 0.10, epochs = 10, batch_size = 5)

    print (history)


scores = model.evaluate(X, y, verbose=0)
print('Accuracy: %.2f%%' % (scores[1]*100))

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Loss')
plt.ylabel('Loss and Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['Accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

import seaborn as sns
from sklearn.metrics import confusion_matrix

def plt_show(img):
    plt.imshow(img)
    plt.show()


cup = 'img_1/10.png'
spoon = 'img_2/10.png'
fork = 'img_3/10.png'
mouse = 'img_4/10.png'

imgs = [cup, spoon, fork, mouse]

classes = None
predicted_classes = [ ]

for i in range(len(imgs)):
    type_ = preprocessing.image.load_img(imgs[i], target_size=(width, height))
    plt.imshow(type_)
    plt.show()

    type_x = np.expand_dims(type_, axis=0)
    prediction = model.predict(type_x)
    index = np.argmax(prediction)
    print(class_names[index])
    classes = class_names[index]
    predicted_classes.append(class_names[index])

cm = confusion_matrix(class_names, predicted_classes)
f = sns.heatmap(cm, xticklabels=class_names, yticklabels=predicted_classes, annot=True)

type_1 = preprocessing.image.load_img(cup, target_size=(witdh, height)) #Need further checking

plt.imshow(type_1)
plt.show()

type_1 = np.expand_dims(type_1, axis=0)
prediction = model.predict(type_1_x)
index = np.argmax(prediction)

print(class_names[index])


type_2 = preprocessing.image.load_img(cup, target_size=(witdh, height)) #Need further checking

plt.imshow(type_2)
plt.show()

type_2 = np.expand_dims(type_2, axis=0)

prediction = model.predict(type_2_x)
                           
index = np.argmax(prediction)
print(class_names[index])


type_3 = preprocessing.image.load_img(cup, target_size=(witdh, height)) #Need further checking

plt.imshow(type_3)
plt.show()

type_3 = np.expand_dims(type_3, axis=0)
prediction = model.predict(type_3_x)

index = np.argmax(prediction)
print(class_names[index])

type_4 = preprocessing.image.load_img(cup, target_size=(witdh, height)) #Need further checking

plt.imshow(type_4)
plt.show()

type_4 = np.expand_dims(type_4, axis=0)
prediction = model.predict(type_4_x)

index = np.argmax(prediction)
print(class_names[index])





#live predicion using camera

from keras.application import inception_v3

import time

CAMERA = cv2.VideoCapture(0)
camera_height  = 500

while (True):
    _, frame = CAMERA.read()

    #flip 
    frame = cv2.flip(frame, 1)

    #rescale the image output
    aspect = frame.shape[1] / float (frame.shape [0])
    res = int (aspect* camera_height)
    frame = cv2.resize (frame, (res, camera_height))

    #get roi

    roi = cv2.cv2Color(roi, cv2.COLOR_BGR2RGB)

#Adjust Alignment 
roi = cv2.resize(roi (width, height))
roi_x = np.expand_dims(roi_axis = 0)

predictions = model.predict(roi_x)
type_1_x, type_2_x, type_3_x, type_4_x = predictions[0]

#The green rectable 
cv2.rectangle(frame, (300,75), (650,425), (240, 100, 0), 2)

#Predictions/Labels
tipe_1_txt ='{} - {}%'.format(class_names[0], int(type_1_x*100))
cv2.putText(frame, tipe_1_txt, (70,210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240,240,240), 2)

tipe_2_txt ='{} - {}%'.format(class_names[1], int(type_2_x*100))
cv2.putText(frame, tipe_2_txt, (70,235), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240,240,240), 2)

tipe_3_txt ='{} - {}%'.format(class_names[2], int(type_3_x*100))
cv2.putText(frame, tipe_3_txt, (70,255), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240,240,240), 2)

tipe_4_txt ='{} - {}%'.format(class_names[3], int(type_4_x*100))
cv2.putText(frame, tipe_4_txt, (70,275), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240,240,240), 2)

cv.imshow("Real time object detection", frame)

#Controls q = quit/s = capturing 
key = cv2.waitKey(1)

if Key & 0xff == ord('q'):
    break

#preview
plt.imshow(frame)
plt.show()


#Camera
CAMERA.release()
cv2.destroyAllWindows()
