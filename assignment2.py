from  ultralytics import  YOLO
import cv2

model = YOLO("yolov5n.pt")
image = [cv2.imread("parking_lot.jpg"), cv2.imread("traffic.jpeg")]

# This will resize the opened image to the given size.
def resize_images(img):
    resized_image = []
    for _ in img:
        resized_images = cv2.resize( _ , (200, 200))
        resized_image.append(resized_images)
    return resized_image

resized_image = resize_images(image)
result = model(resized_image, (200, 200))


for i,r in enumerate(result):
    #r.show()
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

    







