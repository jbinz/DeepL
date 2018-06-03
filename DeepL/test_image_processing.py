import imageProcessing as ip
import cv2

for k in range(0,150):
    string = 'Pi_Pictures/Test/Yoghurt/9_picture'
    string = string + str(k) + '.png'
    batch = ip.resize_to_item(cv2.imread(string, 0), (128,128,1),17)
    cv2.imshow('Output', batch)
    cv2.waitKey(0)