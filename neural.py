import torch
from torch import nn
from torch import optim


def Normalize_by(data):
    min_cut = data - pts_min
    return min_cut / (pts_max - pts_min)


def transformPoint(point):
    p = model(Normalize_by(torch.tensor(point, dtype=torch.float)))
    return (p[0], p[1])


pts_v = [[90,172],[244,189],[812,225],[1405,208],[1548,194],
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

pts_v = torch.tensor(pts_v,dtype=torch.float)
pts_min = pts_v.min()
pts_max = pts_v.max()

model = nn.Sequential(
    nn.Linear(2,1024,bias=True),
    nn.ReLU(),
    nn.Linear(1024,512,bias=True),
    nn.ReLU(),
    nn.Linear(512,2)
)

model.load_state_dict(torch.load("points.pth"))
model.eval()
