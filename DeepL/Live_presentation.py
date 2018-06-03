from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np
from Classification_model import Classification_model as Cm
from Dataset_generator import *
import imageProcessing as ip
import keras.backend as K
from check_equal import *
import os
from collections import Counter

'''
INITIALIZING ALL VARIABLES AND THE MODEL
'''


# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))

# allow the camera to warmup
time.sleep(0.1)

cm = Cm("d(17_5)_s(64)_b(100)_e(400)_r(0.7)_w(0_1)_h(0_1)_z(0_25)_m8")
cm.set_parameter(64,100,400,0.7,0.1,0.1,0.25,8)
cm.load_model()
#cm.train_model()
model = cm.get_model()


no_object_threshold = 0.3
num_class = cm.get_number_of_classes()

#For checking if our prediction is correct just check if the last x frames
#showed the same prediction. if it doesnt then its wrong
precedent =["No Item Detected"]*7
counter = 0
flag = 0
#need to do this for initial value to not always give out the value
preds = 0
labels = ["Ball", "Bottle", "Can", "Cup", "Face", "Pen", "Phone", "Shoe", "Silverware", "Yogurt"]
#If the Pi sees a white image (no object)
noBatch = np.zeros((cm.shape[0],cm.shape[1]))-1;

#Initialising Writing on Image
font = cv2.FONT_HERSHEY_SIMPLEX
beginningText = (10,470)
fontScale = 1
fontColor = (0, 255, 255)
lineType = 2



'''
ACTUAL LOOP, HERE THE NETWORK IS WORKING
'''




for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    #start = time.time()

    image = frame.array
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    batch = ip.resize_to_item(image_gray, cm.shape, 37)
    if(not np.array_equal(batch, noBatch)):
        batch = batch.astype(np.float64)
        batch -= np.mean(batch, keepdims=True)
        batch /= (np.std(batch, keepdims=True) + K.epsilon())
        # predict with the model
        
        preds = model.predict(np.expand_dims(np.expand_dims(batch, axis=0),axis =3))
        #end = time.time()
        #print end - start

        #check if there is a object infront of the camera
        #And also check if the last 5 images seen are the same prediction and
        #if the last prediction is the same then the prediction now
        print precedent
        check = Counter(precedent)
        if check.most_common(1)[0][1] > 3:
        #if checkEqual(precedent):
            if check.most_common(1)[0][0] == "Not sure what Item it is":
                print("Not sure what Item it is")
                cv2.putText(image, 'Not sure what kind of Item',
                        beginningText, font, fontScale,fontColor,lineType)
            elif check.most_common(1)[0][0] == "No Item Detected":
                print("No Item Detected")
                cv2.putText(image, 'No Item Detected',
                        beginningText, font, fontScale,fontColor,lineType)
            else:
                print check.most_common(1)[0][0]
                cv2.putText(image, check.most_common(1)[0][0],
                        beginningText, font, fontScale,fontColor,lineType)
        else:
            print("Not sure what Item it is")
            cv2.putText(image, 'Not sure what kind of Item',
                    beginningText, font, fontScale,fontColor,lineType)


        if np.max(preds) > no_object_threshold:
            precedent[counter] = labels[np.argmax(preds).astype(np.int)]
        else:
            precedent[counter] = "Not sure what Item it is"
    else:
        precedent[counter] = "No Item Detected"
        print("No Item Detected")
        cv2.putText(image, 'No Item Detected',
                beginningText, font, fontScale,fontColor,lineType)

    counter = counter + 1
    if counter == 7:
        counter = 0
    
    key = cv2.waitKey(100)
    if key==ord('q'):
        break
    cv2.imshow('frame', image)

    rawCapture.truncate(0)
