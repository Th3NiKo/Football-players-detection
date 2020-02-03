import cv2
import numpy as np
import os


#Crop enviroment
def crop_frame(frame):
    #Bounds
    pts = np.array([[126,634], [935,690],[1770,655], [1280,460],[581,450]])
    rect = cv2.boundingRect(pts)
    x,y,w,h = rect
    croped = frame[y:y+h, x:x+w].copy()
    #Mask
    pts = pts - pts.min(axis=0)
    mask = np.zeros(croped.shape[:2], np.uint8)
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
    dst = cv2.bitwise_and(croped, croped, mask=mask)
    return dst


#Detector for contours
def create_blob_detector(roi_size=(22, 22), blob_min_area=15, 
                         blob_min_int=.5, blob_max_int=.95, blob_th_step=10,circle=False):
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = blob_min_area
    params.maxArea = roi_size[0]*roi_size[1]
    params.filterByCircularity = circle
    params.minCircularity = 0.7
    params.maxCircularity = 1.1
    params.filterByColor = False
    params.blobColor = 255
    params.filterByConvexity = False
    params.minConvexity = 0.1
    params.maxConvexity = 0.9
    params.minDistBetweenBlobs = 0
    params.filterByInertia = False
    # blob detection only works with "uint8" images.
    params.minThreshold = int(blob_min_int*255)
    params.maxThreshold = int(blob_max_int*255)
    params.thresholdStep = blob_th_step
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3:
        return cv2.SimpleBlobDetector(params)
    else:
        return cv2.SimpleBlobDetector_create(params) 
    

#Show actual mouse position on click and change player one
def get_mouse(event,x,y,flags,param):
    global mouseX,mouseY
    if event == cv2.EVENT_LBUTTONDOWN:
        mouseX,mouseY = x,y
        param.x = x
        param.y = y
        print(str(mouseX) + " " + str(mouseY))


def apply_morph(img, morph_type=cv2.MORPH_CLOSE, kernel_size=(3,3), make_gaussian=False):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    if make_gaussian:
        img = cv2.GaussianBlur(img,(3,3),0)
    return cv2.morphologyEx(img, morph_type, kernel)

def get_contours(img):
    contours,hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_TC89_KCOS  )
    return contours

def draw_contours(img,contours):
    for c in contours:
        cv2.drawContours(img,[c], 0, (255,255,0), 2)

#Which frame to use as countour finder, which frame to use as image output, min area, max area, frame counter
def extract_players(frame,img,min_area, max_area, counter):
    im_count = 0
    if not os.path.exists("extraction"):
        os.mkdir("extraction")
    frame1 = apply_morph(frame,morph_type=cv2.MORPH_DILATE, kernel_size=(3,3), make_gaussian=True)
    contours = get_contours(frame1)
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        if h>=1.1*w:
            if h * w > min_area and h * w < max_area:
                player_img = img[y:y+h,x:x+w]
                name = str(counter) + "_" + str(im_count) + "_" + str(x) +  "_" + str(y) + "_" + str(w) + "_" + str(h)+".jpg"
                cv2.imwrite('extraction/'+name, player_img)
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),3)
                im_count += 1



#Create statics

#Substractor for static background mask
backSub = cv2.createBackgroundSubtractorKNN(detectShadows=False)

#Basic player detector 
detector = create_blob_detector()

#Set HSV colors for each team
lower_team1 = np.array([0,0, 0])
upper_team1 = np.array([34, 256, 256])

lower_team2 = np.array([35,2, 130])
upper_team2 = np.array([110, 140, 255])