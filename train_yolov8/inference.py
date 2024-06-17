from ultralytics import YOLO
import time

import cv2

model = YOLO('train/weights/best.pt')

img_source = 'test_images/10_missing_hole_01.jpg'

[results] = model(img_source, save=False)


print("This is result boxes")
print(results.boxes)         # Bounding box coordinate values in xyxy format 
print("This is confidence values")
print(results.boxes.conf)    # Confidence values for each bounding box prediction 
print(results.boxes.cls)     # Classification for each of the bounding box predicted 

print('NINGAA IS KING')
print(type(results.boxes.cls.tolist()[0]))

sample_img = cv2.imread(img_source)
sample_img_cpy = sample_img.copy()
img_w, img_h = sample_img.shape[1], sample_img.shape[0]

for i in range(len(results.boxes)):
    center_x = results.boxes.xywhn[i][0]
    center_y = results.boxes.xywhn[i][1]
    defect_w = results.boxes.xywhn[i][2]
    defect_h = results.boxes.xywhn[i][3]

    l = int((center_x - defect_w/2)* img_w)
    r = int((center_x + defect_w/2)* img_w)
    t = int((center_y - defect_h/2)* img_h)
    b = int((center_y + defect_h/2)* img_h)

    cv2.rectangle(sample_img_cpy, (l,t), (r,b), (0,255,255), 2) 

    alpha = 0.5
    cool_img = cv2.addWeighted(sample_img_cpy, alpha, sample_img, 1 - alpha, 0)

cool_img = cv2.resize(cool_img, (640,640), interpolation=cv2.INTER_LANCZOS4)
cv2.imshow('hello', cool_img)
cv2.waitKey(0)