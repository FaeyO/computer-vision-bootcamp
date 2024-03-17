from  ultralytics import  YOLO
import cv2

model = YOLO("yolov5n.pt")
image = cv2.imread("office2.jpg")

result = model(image)
#print(result)


for i,r in enumerate(result):
    #r.show()
    #print(r.boxes)
# location of object detected
    detection = r.boxes.data.tolist()
    #print(detection)

#to get the names of what yolo was train with
    names = r.names
#values of object detected
    classes = r.boxes.cls.tolist()
    #print(classes)

    for labels,detection in zip(classes,detection):

        label = names[labels]
        x,y,w,h,conf,_ = detection

#print confidence interval
        #print(conf)

        cv2.rectangle(image,(int(x),int(y)),(int(w),int(h)), (255,0,0),2)
        cv2.putText(image, str(label), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(image, str(round(conf,2)), (int(w), int(h)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        

cv2.imshow("image",image)
cv2.waitKey(0)

    







