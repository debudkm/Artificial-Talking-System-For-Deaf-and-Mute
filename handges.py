# Imports
import cv2
import numpy as np
import math

# open camera
capture = cv2.VideoCapture(0)

while capture.isOpened():
    #capture frames from the camera
    ret, frame = capture.read()

    #get hand data from the rectangle sub window
    cv2.rectangle(frame, (100, 100), (300, 300), (0, 225, 0), 0)
    crop_image = frame[100:300, 100:300]

    #apply Gaussian blur
    blur = cv2.GaussianBlur(crop_image, (3, 3), 0)

    #change color space from BGR->HSV
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    #create a binary image with where white will be skin color and rest is black
    mask2 = cv2.inRange(hsv, np.array([2, 0, 0]), np.array([20, 255, 255]))

    #kernel for morphological transformation
    kernel = np.ones((5, 5))

    #apply morphological transformations to filter out the background noise
    dilation = cv2.dilate(mask2, kernel, iterations=1)
    erosion = cv2.erode(dilation, kernel, iterations=1)

    #apply gaussian blur and threshold
    filtered = cv2.GaussianBlur(erosion, (3, 3), 0)
    ret, thresh = cv2.threshold(filtered, 127, 255, 0)

    #show threshold image
    cv2.imshow("Threshold", thresh)

    #find contours
    image = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = image if len(image) == 2 else image[1:3]

    try:
        #find contour with maximum area
        contour = max(contours, key=lambda x: cv2.contourArea(x))

        #create bounding rectangle around the contour
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(crop_image, (x, y), (x + w, y + h), (0, 0, 255), 0)

        #find convex hull
        hull = cv2.convexHull(contour)

        #draw contour
        drawing = np.zeros(crop_image.shape, np.uint8)
        cv2.drawContours(drawing, [contour], -1, (0, 255, 0), 0)
        cv2.drawContours(drawing, [hull], -1, (0, 0, 255), 0)

        #find convexity defect
        hull = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour, hull)

        #use cosine rule to find angle of the far point from the start and end point i.e.  for all defects
        count_defects = 0

        for i in range(defects.shape[0]):
            s, e, f, d = defects[1, 0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])

            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            angle = (math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180) / 3.14

            #if angle > 90 draw a circle at the far point
            if angle <= 90:
                count_defects += 1
                cv2.circle(crop_image, far, 1, [0, 0, 255], -1)

            cv2.line(crop_image, start, end, [0, 255, 0], 2)

            #print no. of fingers
            if count_defects == 0:
                cv2.putText(frame, "One", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
            elif count_defects == 1:
                cv2.putText(frame, "Two", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
            elif count_defects == 2:
                cv2.putText(frame, "Three", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
            elif count_defects == 3:
                cv2.putText(frame, "Four", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
            elif count_defects == 4:
                cv2.putText(frame, "Five", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
            else:
                pass
    except:
        pass

    #show required images
    cv2.imshow("Gesture", frame)
    all_image = np.hstack((drawing, crop_image))
    cv2.imshow('Contours', all_image)
    #close the camera if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()








