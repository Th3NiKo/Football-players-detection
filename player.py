import kalman
import math
import cv2
import neural

class Player:

    def __init__(self,x,y,label,team):
        self.label = label
        self.x = x
        self.y = y
        self.isAlreadyLabaled = True
        self.team = team #Which team 1 or 2
        self.kalmanFilter = kalman.KalmanFilter()
        self.kalmanFilter.updatePosition((self.x,self.y))
        self.historyRemember = 200
        self.historyPoints = []
        self.isAlreadyLabaled = False

    def updatePosition(self,newx, newy):
        self.historyPoints.append(self.getPoint2F())
        self.x = newx
        self.y = newy
        self.isAlreadyLabaled = True
        if len(self.historyPoints) > self.historyRemember:
            self.historyPoints.pop(0)

    def getPoint2F(self):
        return (self.x,self.y)

    def updateToNearest(self,keypoints,actual_frame):
        actualNearest = (0,0)
        actualDistance = 99999
        allDistances = []
        for key in keypoints:
            distance = math.sqrt(math.pow(key.pt[0] - self.x,2) + math.pow(key.pt[1] - self.y,2))
            if distance < actualDistance:
                actualDistance = distance
                actualNearest = key.pt
            allDistances.append(key.pt)
        #print(actualDistance)
        if actualDistance < 7:
            self.updatePosition(int(actualNearest[0]),int(actualNearest[1]))
            self.kalmanFilter.predictPosition()
            self.kalmanFilter.updatePosition([self.x, self.y])
            self.isAlreadyLabaled = True
        else:
            self.isAlreadyLabaled = False
            if actual_frame > 100:
                predicted = self.kalmanFilter.predictPosition()
                #self.updatePosition(int(predicted[0]),int(predicted[1]))
            else:
                self.tryFindPos(allDistances)

    def drawPlayerAsCircle(self,img):
        cv2.circle(img,self.getPoint2F(), 5, (255,255,255), -1)
        cv2.putText(img,self.label, (self.x-25, self.y-10),  cv2.FONT_HERSHEY_SIMPLEX, 0.3,(255,255,255))

    def drawPlayerOnPitch(self,img):

        point2d = neural.transformPoint(self.getPoint2F())
        x = int(point2d[0].item())
        y = int(point2d[1].item())
        cv2.circle(img,(x,y) , 8, (50,255,0), -1)
        cv2.putText(img,self.label, (x-25, y-10),  cv2.FONT_HERSHEY_SIMPLEX, 0.3,(50,255,0))

    def drawPlayerHistory(self,video_img, pitch_img):
        for pt in self.historyPoints:
            cv2.circle(video_img,pt, 1, (255,255,255), -1)
            point2t = neural.transformPoint(pt)
            cv2.circle(pitch_img,point2t, 1, (50,255,0), -1)


    #Try to find using history
    def tryFindPos(self,distances):
        #Loop through history and find where player was going
        length = len(self.historyPoints)
        directions = []
        xsum = 0
        ysum = 0
        count = 0
        for point1 in range(0,length-1):
            xdir = self.historyPoints[point1+1][0] - self.historyPoints[point1][0]
            ydir = self.historyPoints[point1+1][1] - self.historyPoints[point1][1]
            xsum += xdir
            ysum += ydir
            count += 1.0
        #Get average dir
        if count != 0:
            avgx = xsum / count
            avgy = ysum / count
        else:
            avgx = 0
            avgy = 0

        #Include direction in distance choice + alpha
        alpha = 4
        newx = self.x + avgx * alpha
        newy = self.y + avgy * alpha

        #Check distances again
        actualNearest = (0,0)
        actualDistance = 99999
        for point in distances:
            distance =  math.sqrt(math.pow(point[0] - newx,2) + math.pow(point[1] - newy,2))
            if distance < actualDistance:
                    actualDistance = distance
                    actualNearest = point
        #print(actualDistance)
        if actualDistance < 25:
            self.updatePosition(int(actualNearest[0]),int(actualNearest[1]))
            self.isAlreadyLabaled = True
        else:
            self.isAlreadyLabaled = False


#Try to update positions of all players in given list
def updateAllPlayers(playersList,keypointsTeam1,keypointsTeam2,actual_frame):
    for player in playersList:
        if player.team == 1:
            player.updateToNearest(keypointsTeam1, actual_frame)
        elif player.team == 2:
            player.updateToNearest(keypointsTeam2, actual_frame)
        else:
            print("Team does not exist for player: " + str(player.label))

#Draw all players in given list on video and 2D pitch
def drawAllPlayers(playerList,video_img, pitch_img):
    for player in playerList:
        player.drawPlayerAsCircle(video_img)
        player.drawPlayerOnPitch(pitch_img)
        #player.drawPlayerHistory(video_img, pitch_img)
