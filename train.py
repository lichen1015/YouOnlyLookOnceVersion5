from torch import nn
import torch
import math
from torch import optim


def one_cycle(y1=0.0, y2=1.0, steps=100):
    # lambda function for sinusoidal ramp from y1 to y2
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1


epochs = 30
# Optimizer
batch_size = 64
# adam 或者 SGD
# lr : 0.001 momentum 0.9
# optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
# scheduler
# lf = one_cycle(1, 0.2, epochs)  # cosine 1->hyp['lrf']
# scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)


# ema
