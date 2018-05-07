import cv2
import numpy as np

body_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')

MIN_MATCH_COUNT = 10
detector = cv2.xfeatures2d.SIFT_create(1500)
FLANN_INDEX_KDITREE = 0
flannParam = dict(algorithm=FLANN_INDEX_KDITREE, tree=5)
flann = cv2.FlannBasedMatcher(flannParam, {})
trainImg = cv2.imread("utd.png", 0)
trainKP, trainDesc = detector.detectAndCompute(trainImg, None)
bookReal = 10.5

def detectLogo(QueryImgBGR):
    flag = True
    QueryImg = cv2.cvtColor(QueryImgBGR, cv2.COLOR_BGR2GRAY)
    queryKP, queryDesc = detector.detectAndCompute(QueryImg, None)
    matches = flann.knnMatch(queryDesc, trainDesc, k=2)

    goodMatch = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            goodMatch.append(m)
    if len(goodMatch) > MIN_MATCH_COUNT:
        tp = []
        qp = []
        for m in goodMatch:
            tp.append(trainKP[m.trainIdx].pt)
            qp.append(queryKP[m.queryIdx].pt)
        tp, qp = np.float32((tp, qp))
        H, status = cv2.findHomography(tp, qp, cv2.RANSAC, 3.0)
        h, w = trainImg.shape
        trainBorder = np.float32([[[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]])
        queryBorder = cv2.perspectiveTransform(trainBorder, H)
        cv2.polylines(QueryImgBGR, [np.int32(queryBorder)], True, (0, 255, 0), 3)
        flag = True
    else:
        #print("Not Enough match found- %d/%d" % (len(goodMatch), MIN_MATCH_COUNT))
        flag = False
    return flag

def detectObject(img):
    blueLower = np.array([100, 150, 46])
    blueUpper = np.array([124, 255, 255])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, blueLower, blueUpper)
    bookContour = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
    if len(bookContour) > 0:
        maxArea = max(bookContour, key=cv2.contourArea)
        x,y,w,h = cv2.boundingRect(maxArea)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        return h
    else:
        return 0

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        body = body_cascade.detectMultiScale(gray, 1.05, 3)
        bodyMaxHeight = 0
        bodyMaxArray = [0,0,0,0]
        ifBody = False
        for (bx,by,bw,bh) in body:
            ifBody = True
            body_gray = gray[by:by+bh, bx:bx+bw]
            body_color = img[by:by+bh, bx:bx+bw]
            flag = detectLogo(body_color)
            if(flag):
                if bh > bodyMaxHeight:
                    bodyMaxHeight = bh
                    bodyMaxArray = [bx,by,bw,bh]
        if(ifBody):
            cv2.rectangle(img, (bodyMaxArray[0],bodyMaxArray[1]), (bodyMaxArray[0]+bodyMaxArray[2],bodyMaxArray[1]+bodyMaxArray[3]), (0, 0, 255), 2)
            body_gray = gray[bodyMaxArray[1]:bodyMaxArray[1]+bodyMaxArray[3],bodyMaxArray[0]:bodyMaxArray[0]+bodyMaxArray[2]]
            body_color = img[bodyMaxArray[1]:bodyMaxArray[1]+bodyMaxArray[3],bodyMaxArray[0]:bodyMaxArray[0]+bodyMaxArray[2]]
            #bookPic/bookReal = personPic/personReal  ==> personReal = bookReal*personPic/bookPic
            objectHeight = detectObject(img)
            result = ""
            if objectHeight != 0:
                result = bookReal * bodyMaxArray[3] / objectHeight

            text = "person height is " + str(result)
            cv2.putText(img, text, (bodyMaxArray[0],bodyMaxArray[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            face = face_cascade.detectMultiScale(body_gray)
            for (fx,fy,fw,fh) in face:
                cv2.rectangle(body_color, (fx,fy), (fx+fw,fy+fh), (255, 0, 0), 2)
                cv2.putText(body_color, 'face', (fx, fy - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                face_gray = gray[fy:fy+fh, fx:fx+fw]
                face_color = img[fy:fy+fh, fx:fx+fw]
                eye = eye_cascade.detectMultiScale(face_gray)
                for (ex,ey,ew,eh) in eye:
                    cv2.rectangle(face_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                    cv2.putText(face_color,'eye',(ex,ey - 20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

        cv2.imshow('img',img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
