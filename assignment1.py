from  ultralytics import  YOLO
import cv2

#model = YOLO("yolov5n.pt")
model = YOLO("yolov5nu.pt") 
image = cv2.imread("people.jpg")

result = model(image)



for i,r in enumerate(result):
# location of object detected
    detection = r.boxes.data.tolist()
#to get the names of what yolo was train with
    names = r.names
#values of object detected
    classes = r.boxes.cls.tolist()

    for labels,detection in zip(classes,detection):

        label = names[labels]
        x,y,w,h,conf,_ = detection

        cv2.rectangle(image,(int(x),int(y)),(int(w),int(h)), (255,0,0),2)
        cv2.putText(image, str(label), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(image, str(round(conf,2)), (int(w), int(h)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        

cv2.imshow("image",image)
cv2.waitKey(0)

    







