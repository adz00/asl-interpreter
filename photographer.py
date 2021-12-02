import cv2

videoCaptureObject = cv2.VideoCapture(0)
ctr = 0
while(True):
    ret,frame = videoCaptureObject.read()
    cv2.imshow('Capturing Video',frame)
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        cv2.imwrite("./customdata/del/del_"+str(ctr)+".jpg", frame)
        ctr += 1
    if(ctr == 500):
        videoCaptureObject.release()
        cv2.destroyAllWindows()
        break