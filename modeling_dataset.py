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