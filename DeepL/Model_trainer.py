import cv2
import numpy as np
from Classification_model import Classification_model as Cm


# Script for training different models

cm3 = Cm("d(18_5)_s(64)_b(100)_e(400)_r(0.7)_w(0_1)_h(0_1)_z(0_25)_m10")
cm3.set_parameter(64,100,400,0.7,0.1,0.1,0.25,10)
cm3.load_model()
cm3.train_model()


cm2 = Cm("d(18_5)_s(64)_b(100)_e(400)_r(0.7)_w(0_1)_h(0_1)_z(0_25)_m9")
cm2.set_parameter(64,100,400,0.7,0.1,0.1,0.25,9)
cm2.load_model()
cm2.train_model()

