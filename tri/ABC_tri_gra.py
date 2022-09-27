# # 这是转角双层石墨烯的连续模型和紧束缚模型的代码整理，里面包含了一些功能实现，如吸收谱，拉曼光谱等。
import numpy as np
import cmath
import multiprocessing
import os.path
import time
from multiprocessing import Pool

import numpy as np
from numpy import exp, sin, array, conj, linspace, real, arange, dot, save, zeros, imag, ones, average, sqrt, pi, cos
# basic parameters
from numpy.linalg import norm, eig
import cv2 as cv
import shutil
import matplotlib
# matplotlib.use("agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# # 这部分为在计算当中需要用到的常数
from public.consts import *


class AbcTriGra:
    pass


