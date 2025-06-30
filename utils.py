import numpy as np
import re
from matplotlib import pyplot as plt


class NestedExit(Exception): pass


def setFigPos(x, y=0, w=200, h=200):
	if isinstance(x, str):
		x, y, w, h = map(lambda el: int(el)//2, re.findall(r"\d+", x))
	plt.get_current_fig_manager().window.setGeometry(x, y, w, h)


def close_figure(event):
	if event.key == "escape":
		plt.close(event.canvas.figure)


def transformToUnitSquarePadded(x, unitSquarePerc):
	xb1 = np.min(x[0, :])
	yb1 = np.min(x[1, :])
	xb2 = np.max(x[0, :])
	yb2 = np.max(x[1, :])
	xr = xb2 - xb1
	yr = yb2 - yb1
	rang = max(xr, yr) / unitSquarePerc
	offset = (1 - unitSquarePerc)/2
	x[0, :] = (x[0, :] - xb1)/rang + ((unitSquarePerc - xr/rang)/2 + offset)
	x[1, :] = (x[1, :] - yb1)/rang + ((unitSquarePerc - yr/rang)/2 + offset)
