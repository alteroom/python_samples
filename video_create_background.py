import cv2
import numpy as np

input_file = "video/vtest.mp4"
#"E:\Signa_video\test\vtest.avi"

video = cv2.VideoCapture(input_file)

print(video.get(cv2.CAP_PROP_FRAME_COUNT))
print(video.get(cv2.CAP_PROP_POS_FRAMES))
video.read()
video.set(cv2.CAP_PROP_POS_FRAMES, 1900)
print(video.get(cv2.CAP_PROP_POS_FRAMES))


bg_sub = cv2.createBackgroundSubtractorMOG2(
    history=1000,
    varThreshold=100,
    detectShadows=False)

i = 0
while True:
    success, frame = video.read()
    if not success:
        break
    
    fg_mask = bg_sub.apply(frame)
    
    cv2.imshow("Video", frame)
    #cv2.imshow("MASK", fg_mask)
    
    if cv2.waitKey(1) == ord('q'):
        break

bg = bg_sub.getBackgroundImage()
#cv2.imwrite("images/bg.jpeg", bg)
#cv2.imshow("BG", bg)
cv2.waitKey()



