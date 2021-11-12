#!python3

import cv2
import numpy as np
import glob
import math


def nothing(x):
    pass


def change_img(k, i):
    p = 121 - k
    i += p
    if i == -1:
        i = len(imgs)-1
    elif i == len(imgs):
        i = 0
    return i, imgs[i]


imgs = glob.glob('*.jpg')

i = 0
img = cv2.imread(imgs[0])
x = img.shape[1]
y = img.shape[0]


cv2.namedWindow('Image')
cv2.namedWindow('Bars')
cv2.resizeWindow('Bars', 275, 350)
cv2.createTrackbar('H-high', 'Bars', 179, 179, nothing)
cv2.createTrackbar('H-low', 'Bars', 0, 179, nothing)
cv2.createTrackbar('S-high', 'Bars', 255, 255, nothing)
cv2.createTrackbar('S-low', 'Bars', 80, 255, nothing)
cv2.createTrackbar('V-high', 'Bars', 255, 255, nothing)
cv2.createTrackbar('V-low', 'Bars', 120, 255, nothing)
cv2.createTrackbar('Height', 'Bars', 40, 100, nothing)
cv2.createTrackbar('Width', 'Bars', 40, 100, nothing)


while True:
    Hue_low = cv2.getTrackbarPos('H-low', 'Bars')
    Hue_high = cv2.getTrackbarPos('H-high', 'Bars')
    Sat_low = cv2.getTrackbarPos('S-low', 'Bars')
    Sat_high = cv2.getTrackbarPos('S-high', 'Bars')
    Val_low = cv2.getTrackbarPos('V-low', 'Bars')
    Val_high = cv2.getTrackbarPos('V-high', 'Bars')
    height = cv2.getTrackbarPos('Height', 'Bars')/100
    width = cv2.getTrackbarPos('Width', 'Bars')/100

    xmin = round((x-width*x)/2)
    ymin = round((y-height*y)/2)
    xmax = x-xmin
    ymax = y-ymin

    r_img = img[ymin:ymax, xmin:xmax]

    img_HSV = cv2.cvtColor(r_img, cv2.COLOR_BGR2HSV)
    dim = (r_img.shape[1], r_img.shape[0])
    resized_dim = tuple([round(i*0.5) for i in dim])
    # img = cv2.resize(img,resized_dim)

    HSV_low = np.array([Hue_low, Sat_low, Val_low])
    HSV_high = np.array([Hue_high, Sat_high, Val_high])

    dot_mask = cv2.inRange(img_HSV, HSV_low, HSV_high)
    result = cv2.bitwise_and(r_img, r_img, mask=dot_mask)

    gray = cv2.split(result)[2]
    canny = cv2.Canny(gray, 200, 300)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    contours, hier = cv2.findContours(blur, 1, 2)
    try:

        cnt = contours[0]
        el = cv2.fitEllipse(cnt)
        el_area = math.pi * el[1][0] * el[1][1]
        # print(el_area)
        z = np.zeros((dim[1], dim[0]), dtype=np.uint8)
        cv2.ellipse(z, el, color=255, thickness=-1)
        #cv2.imshow('el', z)
        el_mask = cv2.bitwise_and(result, result, mask=z)
        # cv2.imshow('el', el_mask)
        pixels = np.count_nonzero(el_mask)
        per = round((pixels/el_area)*100, 1)
        # circle stuff
        # (j, k), radius = cv2.minEnclosingCircle(cnt)
        # center = (int(j), int(k))
        # radius = int(radius))

        cv2.ellipse(result, el, (0, 255, 0), 1)
        cv2.putText(result, str(per), (50, 50), cv2.FONT_HERSHEY_COMPLEX,
                    1, (255, 255, 150), 2, cv2.LINE_AA)
    except:
        pass

    result_small = cv2.resize(result, resized_dim)
    cv2.imshow('Image', result)

    k = cv2.waitKey(1)
    if k == 27:
        break
    elif k == 120 or k == 122:
        i, img = change_img(k, i)
        img = cv2.imread(img)
        x = img.shape[1]
        y = img.shape[0]

cv2.destroyAllWindows()
