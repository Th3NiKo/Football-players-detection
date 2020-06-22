#Imports
import cv2
#import matplotlib.pyplot as plt
import numpy as np
import colorsys
import imutils
from imutils.object_detection import non_max_suppression
from imutils import paths

#My scripts
import player
import neural
import vision

#Load resources
video = cv2.VideoCapture("low3.mp4")
pitch = cv2.imread('field.png',1)

#Label on players and set their initial position
player1 = player.Player(565,104,"Zawodnik 1",1)
player2 = player.Player(875,52,"Zawodnik 2",1)
player3 = player.Player(728,101,"Zawodnik 3",1)

#Prepare mouse events
cv2.namedWindow('Video')
cv2.setMouseCallback('Video',vision.get_mouse,param=player1)

allPlayers = [player1,player2,player3]
#allPlayers = [player1]

#Create writers for output to files
out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (1644,240))
outpitch = cv2.VideoWriter('pitch.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (800,518))

count = 0
while True:
    count = count + 1 #frame count
    ret,frame = video.read()
    #a = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    if not frame is None :

        #Crop image and remove static background
        frame = vision.crop_frame(frame)

        fgMask = vision.backSub.apply(frame)

        #Get masks for each team
        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, vision.lower_team1, vision.upper_team1)
        mask2 = cv2.inRange(hsv, vision.lower_team2, vision.upper_team2)

        #Do masking
        res = cv2.bitwise_and(fgMask, fgMask, mask=mask)

        res2 = cv2.bitwise_and(fgMask, fgMask, mask=mask2)

        #Do Morphologic


        #Detect blobs for each team
        keypointsTeam1 = vision.detector.detect(res)
        keypointsTeam2 = vision.detector.detect(res2)

        #Try to update player positions
        player.updateAllPlayers(allPlayers,keypointsTeam1,keypointsTeam2,count)

        #Draw players actual positions
        pitchTemp = cv2.cvtColor(pitch.copy(),cv2.COLOR_BGR2RGB)
        player.drawAllPlayers(allPlayers, frame, pitchTemp)


        #frame = cv2.drawKeypoints(frame, keypointsTeam1, np.array([]),(255,0,0), cv2.DRAW_MATCHES_FLAGS_DEFAULT)
        #frame = cv2.drawKeypoints(frame, keypointsTeam2, np.array([]),(0,0,255), cv2.DRAW_MATCHES_FLAGS_DEFAULT)

        '''
        pts = np.array([[126,634], [935,690],[1770,655], [1280,460],[581,450]])
        rect = pts.reshape(pts.shape[0],1,pts.shape[1])
        for x in range(0,1645):
            for y in range(185,220):#238
                r = x / 1645 * 255
                g = y / 238 * 255
                b = (x + y)/ (1645 + 238) * 255
                frame = cv2.line(frame, (x,y), (x,y), (r, g, b), 1)
                newpt = neural.transformPoint((x,y))
                pitchTemp = cv2.line(pitchTemp, newpt, newpt, (r, g, b), 1)

        if count > 100 and count % 25 == 0:
            vision.extract_players(fgMask,frame,150,420,count)
        '''
        cv2.imshow('Video', frame)
        #out.write(frame)

        cv2.imshow('Pitch',pitchTemp)
        #outpitch.write(pitchTemp)

        keyboard = cv2.waitKey(20) & 0xFF
        if keyboard == 'q' or keyboard == '27':
            break

out.release()
outpitch.release()
