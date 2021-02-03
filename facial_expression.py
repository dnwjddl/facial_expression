# usage
# python facial_expression.py --cascade haarcascade_frontalface_default.xml --model resnet_model_filter.h5

from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-c', '--cascade', required = True)
ap.add_argument('-m', '--model', required = True)
ap.add_argument('-v', '--video')
args = vars(ap.parse_args())

#load face detector cascade
detector = cv2.CascadeClassifier(args['cascade'])
model = load_model(args['model'])
EMOTIONS = ['angry', 'scared', 'happy', 'sad', 'surprised', 'neutral']

if not args.get('video', False):
    camera = cv2.VideoCapture(0)
else:
    camera = cv2.VideoCapture(args['video'])

while True:
    # camera status, frame
    ## if ret is True, 정상작동
    (ret, frame) = camera.read()

    if args.get('video') and not ret:
        break

    # resize the frame and convert it to grayscale
    frame = imutils.resize(frame, width=700)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #initialize the canvas for the visualization, clone
    # the frame so we can draw on it
    # 왜 크기가 저거지
    canvas = np.zeros((48, 48, 3), dtype= 'uint8')
    frameClone = frame.copy()

    # Cascade Classifier의 detectMultiScale 함수에 grayscale 이미지를 입력하여 얼굴을 검출
    # 얼굴이 검출되면 위치를 리스트로 리턴 #(x,y,w,h) 튜플형태

    ## scale Factor: 이미지에서 얼굴 크기가 서로 다른 것을 보상해주는 값
    ## minNeighbors: 얼굴 사이의 최소 간격(픽셀)
    ## minsize: 얼굴의 최소 크기
    rects = detector.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 5, minSize = (30, 30), flags = cv2.CASCADE_SCALE_IMAGE)

    # face를 찾았을 경우
    if len(rects) > 0:
        #face area
        rect = sorted(rects, reverse=True, key = lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]

        (fX, fY, fW, fH) = rect

        roi = gray[fY:fY+fH, fX:fX+fW]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype('float')/255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis = 0)

        #preds
        preds = model.predict(roi)[0]
        label = EMOTIONS[preds.argmax()]

    for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
        text = "{}: {:.2f}%".format(emotion, prob*100)

        w= int(prob * 300)
        cv2.rectangle(canvas, (5, (i*35) + 5),
        (w, (i*35)+35), (0,0,225), -1)
        cv2.putText(canvas, text, (10, (i * 35) + 23),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 2)

        cv2.putText(frameClone, label, (fX, fY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,0,0), 2)
        cv2.rectangle(frameClone, (fX, fY), (fX+fW, fY+fH),(255,0,0), 2)

        cv2.imshow("face", frameClone)
        cv2.imshow("prob", canvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 카메라 장치에서 받아온 메모리 해제
camera.release()
# 모든 윈도우 창 닫기
cv2.destroyAllWindows()