import cv2
import torch
import numpy as np
from Face_Recog.ML_FR.train_locator import opencv_get_faces

device = torch.device('cpu')
net = torch.load('computer_club#2.pth',map_location=device)
net.to(device)
video_capture = cv2.VideoCapture('test.mov')
success, frame = video_capture.read()

shape = frame.shape
out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, (shape[1], shape[0]))

classes = ['Ava', 'Clara', 'Dima', 'Lauren', 'Mariana', 'Naomi',
           'Nia','Nikita', 'Oleg', 'Paul', 'Raymond', 'Signe', 'Taira']

ignore = 4
while success:
    if ignore<=0:
        boxes = opencv_get_faces(frame)
        rgb_frame = frame[:,:,::-1].transpose((2,0,1))/255.0
        for box in boxes:
            x1, y1, w, h = box
            if w > 40 and h > 40:
                one_face = rgb_frame[:,y1:y1+h, x1:x1+w]
                input = np.expand_dims(one_face, axis=0)
                input = torch.tensor(input, dtype=torch.float32)
                input = input.to(device)
                output = net(input)
                _, argmax = torch.max(output,1)
                name = classes[argmax]
                frame = cv2.rectangle(frame, (x1, y1, w, h), (0,0,255))
                frame = cv2.putText(frame, name, (box[0] + 6, box[1] - 15),
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0))
        out.write(frame)
    else:
        ignore-=1
    success, frame = video_capture.read()

video_capture.release()
out.release()
cv2.destroyAllWindows()

