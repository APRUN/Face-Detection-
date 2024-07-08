import cv2
obj=cv2.CascadeClassifier("C:/Users/Chief Oggy/AppData/Local/Programs/Python/Python312/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")
vid_cap=cv2.VideoCapture(0) #0 for runtime
while True:
    ret, video_data=vid_cap.read()
    mono_chromed=cv2.cvtColor(video_data,cv2.COLOR_BGR2GRAY)
    faces=obj.detectMultiScale(
        mono_chromed,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30,30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    for (x,y,w,h) in faces:
        cv2.rectangle(video_data,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow("Hacking",video_data)
    if cv2.waitKey(10)==ord("a"):
        break
vid_cap.release()