#!/usr/bin/python3
import cv2
import numpy as np

pts_video = [[90,172],[244,189],[812,225],[1405,208],[1548,194],
                            [1276,86],[1151,34],
                            [1057,32],[822,25],[590,23],[500,24],
                            [821,72],[442,68],[1200,82],[363,124],[1273,138],
                            [1086,47],[559,37],
                            [1125,79],[516,71],
                            [225, 118],[305, 88],[340, 77],[381, 61],[418, 50],[469, 34],
                            [456,84],[518,55],
                            [733,73],[817,96],[820,57],[910,76],
                            [1190,98],[1129,66],
                            [1411,137],[1336,103],[1300,92],[1261,76],[1230,63],[1180,49],
                            [630,74],[1023,73],
                            [469,213],[1159,222],[1083, 148],[558,134]
]
pts_image = [[0,518],[132,518],[400,518],[668,518],[800,518], #Dół planszy od lewej do prawej
                           [800,259],[800,0],#srodek bramki prawo, gorny prawy rog
                           [668,0],[400,0],[132,0],[0,0], #góra od prawej do lewej
                           [400,260],[88,259],[711,259],[132,420],[667,420], #środek, srodek lewy karny, srodek prawy karny, dol karny lewo, dol karny prawo
                           [668,98],[132,98],  #góra karny prawo, góra karny lewo
                           [639,258],[160,258], #polkole prawo wysuniety, pokole lewo wysuniety
                           [0,420],[0,332],[0,280],[0,238],[0,186],[0,98], #lewa strona karne punkty od dolu do goryh
                           [132,316],[132,202], #lewe polkole dol, lewe polkole gora
                           [326,256],[400,332],[400,185],[472,256], #srodek kolo lewo, srodek kolo dol, srodek kolo gora,  srodek kolo prawo
                           [668,315],[668,202], #prawe pokole dol, prawe pokole gora
                           [800,420],[800,332],[800,280],[800,238],[800,186],[800,98], #prawo calosc od dolu do gory karne
                           [244,256],[556,256], #srodek lewo, srodek prawo
                           [266,518],[534,518],[534,420],[266,420]#polowa lewo dol, polowa prawo dol ,polowa lewo miedzysrodek dol, polowa prawo miedzysrodek dol
]


h, status = cv2.findHomography(np.array([pts_video]), np.array([pts_image]))


def transformPoint(pts):
    """
    Function for translating video position to 2D using homography matrix
    
    Input
    ------
    pts : [x, y]
        array containg video x,y point
    
    Output
    ------
    [x', y']
        array containg translated 2D x',y' point
    """
    pts = np.array([[pts]],dtype=np.float32)
    dst = cv2.perspectiveTransform(pts,h)
    return [dst[0,0,0],dst[0,0,1]]