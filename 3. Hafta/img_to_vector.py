import cv2

def img_to_vector(file_name,w,h):
    return cv2.resize(cv2.imread(file_name,0),(w,h)).reshape(1,w*h)[0]
