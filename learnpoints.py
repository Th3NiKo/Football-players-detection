
#!/usr/bin/python3
import torch
from torch import nn
from torch import optim
import math
import random

def Normalize(data):
    min_cut = data - data.min()
    return min_cut / (data.max() - data.min())

def Normalize_by(data,x):
    min_cut = data - x.min()
    return min_cut / (x.max() - x.min())


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


testX = Normalize_by(torch.tensor([[284,160], [1347,178], [818,136]], dtype=torch.float),torch.tensor(pts_video,dtype=torch.float))
testY = torch.tensor([[130,474], [670,474] , [400,422]], dtype=torch.float)


x = Normalize(torch.tensor(pts_video, dtype=torch.float))
y = torch.tensor(pts_image, dtype=torch.float)


model = nn.Sequential(
    nn.Linear(2,1024,bias=True),
    nn.ReLU(),
    nn.Linear(1024,512,bias=True),
    nn.ReLU(),
    nn.Linear(512,2)
)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())


for i in range(100000):
    for iterator in range(0,len(pts_image)):
        optimizer.zero_grad()
        predicted = model(x[iterator])

        loss = criterion(predicted, y[iterator])
        #print(loss)
        loss.backward()
        optimizer.step()
        
    diff = 0
    print(loss)
    for i in range(0,len(testX)):
        model.eval()
        out = model(testX[i])
        diff += (abs(testY[i] - out))
    print("*** DIFF ***")
    print((diff[0] + diff[1]).item() / len(testX))
    print("************")

    torch.save(model.state_dict(), "points.pth")

        


