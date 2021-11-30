#!python3

import cv2
import numpy as np
import glob
import math

drawing = False
ix, iy = -1, -1


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


def draw_circle(event, mouse_loc_x, mouse_loc_y, flags, param):
    global ix, iy, cx, cy, drawing, radi

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = mouse_loc_x, mouse_loc_y
        return mouse_loc_x, mouse_loc_y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            radi = int(math.sqrt(((ix-mouse_loc_x)**2)+((iy-mouse_loc_y)**2)))
            cx, cy = mouse_loc_x, mouse_loc_y

    elif event == cv2.EVENT_LBUTTONUP:
        radi = int(math.sqrt(((ix-mouse_loc_x)**2)+((iy-mouse_loc_y)**2)))
        cx, cy = mouse_loc_x, mouse_loc_y
        drawing = False


imgs = glob.glob('*.jpg')

i = 0
img = cv2.imread(imgs[0])
x = img.shape[1]
y = img.shape[0]


cv2.namedWindow('Image')
cv2.namedWindow('Bars')
cv2.resizeWindow('Bars', 275, 370)
cv2.createTrackbar('H-high', 'Bars', 179, 179, nothing)
cv2.createTrackbar('H-low', 'Bars', 0, 179, nothing)
cv2.createTrackbar('S-high', 'Bars', 255, 255, nothing)
cv2.createTrackbar('S-low', 'Bars', 80, 255, nothing)
cv2.createTrackbar('V-high', 'Bars', 255, 255, nothing)
cv2.createTrackbar('V-low', 'Bars', 120, 255, nothing)
cv2.createTrackbar('Height', 'Bars', 40, 100, nothing)
cv2.createTrackbar('Width', 'Bars', 40, 100, nothing)
cv2.createTrackbar('Cir Method', 'Bars', 0, 2, nothing)


while True:
    Hue_low = cv2.getTrackbarPos('H-low', 'Bars')
    Hue_high = cv2.getTrackbarPos('H-high', 'Bars')
    Sat_low = cv2.getTrackbarPos('S-low', 'Bars')
    Sat_high = cv2.getTrackbarPos('S-high', 'Bars')
    Val_low = cv2.getTrackbarPos('V-low', 'Bars')
    Val_high = cv2.getTrackbarPos('V-high', 'Bars')
    height = cv2.getTrackbarPos('Height', 'Bars')/100
    width = cv2.getTrackbarPos('Width', 'Bars')/100
    method = cv2.getTrackbarPos('Cir Method', 'Bars')

    xmin = round((x-width*x)/2)
    ymin = round((y-height*y)/2)
    xmax = x-xmin
    ymax = y-ymin

    r_img = img[ymin:ymax, xmin:xmax]

    img_HSV = cv2.cvtColor(r_img, cv2.COLOR_BGR2HSV)
    dim = (r_img.shape[1], r_img.shape[0])
    resized_dim = tuple([round(i*0.5) for i in dim])

    HSV_low = np.array([Hue_low, Sat_low, Val_low])
    HSV_high = np.array([Hue_high, Sat_high, Val_high])

    dot_mask = cv2.inRange(img_HSV, HSV_low, HSV_high)
    result = cv2.bitwise_and(r_img, r_img, mask=dot_mask)

    gray = cv2.split(result)[2]
    canny = cv2.Canny(gray, 200, 300)

    if method == 0:
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
    else:
        blur = cv2.GaussianBlur(gray, (11, 11), 0)

    contours, hier = cv2.findContours(blur, 1, 2)
    try:

        z = np.zeros((dim[1], dim[0]), dtype=np.uint8)
        cnt = contours[0]
        if method == 0:  # Detect Ellipse
            el = cv2.fitEllipse(cnt)
            cv2.ellipse(z, el, color=255, thickness=-1)
            cv2.ellipse(result, el, (0, 255, 0), 1)
        elif method == 1:  # Detect Circle
            minDist = 100
            param1 = 200  # 500
            param2 = 40  # 200 #smaller value-> more false circles
            minRadius = 5
            maxRadius = 100  # 10
            circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, minDist,
                                       param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
            circles = np.uint16(np.around(circles))
            for j in circles[0, :]:
                cv2.circle(z, (j[0], j[1]), j[2], 255, -1)
                cv2.circle(result, (j[0], j[1]), j[2], (0, 255, 0), 1)
        elif method == 2:  # Draw Circle
            cv2.setMouseCallback('Image', draw_circle)

            cv2.circle(result, (cx, cy), radi, (0, 255, 0), thickness=1)
            cv2.circle(z, (cx, cy), radi, 255, thickness=-1)

        el_mask = cv2.bitwise_and(result, result, mask=z)
        el_mask_gray = cv2.cvtColor(el_mask, cv2.COLOR_BGR2GRAY)
        pixels = np.count_nonzero(el_mask_gray)
        el_pixels = np.count_nonzero(z)
        per = round((pixels/el_pixels)*100, 1)

        cv2.putText(result, str(per), (50, 50), cv2.FONT_HERSHEY_COMPLEX,
                    1, (255, 255, 150), 2, cv2.LINE_AA)
    except:
        pass

    result_small = cv2.resize(result, resized_dim)
    cv2.imshow('Image', result)
    # cv2.imshow('i', z)
    # cv2.imshow('j', el_mask)
    k = cv2.waitKey(1)
    if k == 27:
        break
    elif k == 120 or k == 122:
        i, img = change_img(k, i)
        img = cv2.imread(img)
        x = img.shape[1]
        y = img.shape[0]

cv2.destroyAllWindows()
