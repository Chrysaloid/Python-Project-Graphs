class NestedExit(Exception): pass

import re
from matplotlib import pyplot as plt
def setFigPos(x, y = 0, w = 200, h = 200):
	if isinstance(x, str):
		x,y,w,h = map(lambda el : int(el)//2, re.findall(r"\d+", x))
	plt.get_current_fig_manager().window.setGeometry(x,y,w,h)

def close_figure(event):
	if event.key == "escape":
		plt.close(event.canvas.figure)
