from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np
from Classification_model import Classification_model as Cm
from Dataset_generator import *
import imageProcessing as ip
import keras.backend as K


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


no_object_threshold = 0.5
num_class = cm.get_number_of_classes()

noBatch = np.zeros((cm.shape[0],cm.shape[1]))-1;

#Initialising Writing on Image
font = cv2.FONT_HERSHEY_SIMPLEX
beginningText = (10,470)
fontScale = 1
fontColor = (0, 255, 255)
lineType = 2



labels = ["Ball", "Bottle", "Can", "Cup", "Face", "Pen", "Phone", "Shoe", "Silverware", "Yogurt"]

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    start = time.time()
    image = frame.array
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    batch = ip.resize_to_item(image_gray, cm.shape, 17)
    cv2.imshow('batch',batch)
    if(not np.array_equal(batch, noBatch)):
        batch = batch.astype(np.float64)
        batch -= np.mean(batch, keepdims=True)
        batch /= (np.std(batch, keepdims=True) + K.epsilon())
        # predict with the model
    
        preds = model.predict(np.expand_dims(np.expand_dims(batch, axis=0),axis =3))
        end = time.time()
        print end - start
        #print preds
        #check if there is a object infront of the camera
        if np.max(preds) > no_object_threshold:
            print labels[np.argmax(preds).astype(np.int)]
            cv2.putText(image, labels[np.argmax(preds).astype(np.int)],
                        beginningText, font, fontScale,fontColor,lineType)
        else:
            print("not sure what kind of item")
            cv2.putText(image, 'not sure what kind of item',
                        beginningText, font, fontScale,fontColor,lineType)
    else:
        print("no item detected")
        cv2.putText(image, 'no item detected',
                    beginningText, font, fontScale,fontColor,lineType)
    key = cv2.waitKey(100)
    if key==ord('q'):
        break

    cv2.imshow('frame', image)

    rawCapture.truncate(0)


