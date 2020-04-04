from keras.models import load_model
import cv2
import numpy as np
#from playsound import playsound
import winsound
winsound.PlaySound("voice.au", winsound.SND_ASYNC | winsound.SND_ALIAS )

model = load_model('model.h5')
#values[list(dict_.keys())[i]]=y_out[0][i]
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

img = cv2.imread('img_1.jpg')
img = cv2.resize(img,(224,224))
img = np.reshape(img,[1,224,224,3])

classes = model.predict(img)

print (classes)
index_max = np.argmax(classes)

count=0
video_capture = cv2.VideoCapture(0)

while True:

    ret, frame = video_capture.read()
    img=cv2.resize(frame,(224,224))
    img=np.reshape(img,[1,224,224,3])
    classes=model.predict(img)
    # print("here",classes)
    # print("here",classes[0][0])
    if(int(classes[0][0])>0.7):
        count+=1
        cv2.putText(frame, "Avoid Face Touching -:{}".format(count),(50,50),cv2.FONT_ITALIC, 1, (0, 255, 0), 3)
        winsound.PlaySound("voice.au", winsound.SND_ASYNC | winsound.SND_ALIAS )
        winsound.PlaySound(None, winsound.SND_ASYNC)
        #cv2.putText(frame, "no.of times touched the face:{}".format(count),(70,60),cv2.FONT_ITALIC, 1, (0, 255, 0), 3)
        #playsound('voice.m4a')
        #winsound.PlaySound("voice.wav", winsound.SND_FILENAME)
        #import time
        #time.sleep(1.5)

        cv2.imshow('hand detected',frame)

        cv2.namedWindow("hand detected",cv2.WINDOW_NORMAL)
        cv2.waitKey(800)
        cv2.destroyAllWindows()

    else:
    	print("no hand detected")

    #print(classes[0])

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
