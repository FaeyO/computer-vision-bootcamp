import cv2 as cv 


cap = cv.VideoCapture("bgr.mp4")

fourcc = cv.VideoWriter_fourcc(*'mp4v')
output = cv.VideoWriter('out.mp4',fourcc,30,(400,200))

while cap.isOpened:
    ret, frame  = cap.read()

    # if not ret:
    #     break

    if not ret:
        cap = cv.VideoCapture("bgr.mp4")
        continue

    frame = cv.resize(frame,(400,200))
    vid_Gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    blur_Vid = cv.GaussianBlur(frame,(7,7),0)
    vidBlur = cv.blur(frame,(7,7))
    _,thresh_frame = cv.threshold(vid_Gray,50,150,cv.THRESH_BINARY)
    

    output.write(frame)
    #output.write(blur_Vid)

    cv.imshow('frame',frame)
    cv.imshow('frame_GRAY',vid_Gray)
    cv.imshow('frame_BLUR',blur_Vid)
    cv.imshow('frame_BLUR2',vidBlur)
    cv.imshow('frame_THRESH',thresh_frame)
    

    k = cv.waitKey(1) #& 0xFF

    if k == 27:
        break

cap.release()
output.release()
cv.destroyAllWindows()