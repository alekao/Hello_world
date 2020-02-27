import cv2
from mtcnn.mtcnn import MTCNN
from cv2 import CascadeClassifier

clf = CascadeClassifier('haarcascade_frontalface_default.xml')
def opencv_get_faces(frame):
    boxes = clf.detectMultiScale(frame, minNeighbors=8)
    return boxes


model = MTCNN()
def mtcnn_get_faces(frame):
    faces = model.detect_faces(frame)
    boxes = []
    for face in faces:
        boxes.append(face['box'])
    return boxes


