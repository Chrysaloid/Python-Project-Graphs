import numpy as np
from matplotlib import pyplot as plt
from utils import close_figure, setFigPos
from PchipInterpolatorClip import PchipInterpolatorClip
from scipy.interpolate import PchipInterpolator, interp1d

plt.style.use("dark_background")

fig, ax = plt.subplots(tight_layout=True)

# func = interp1d([0, 1, 2, 3], [2, 1, 4, 5], kind="linear", fill_value="extrapolate", copy=False)
# func = PchipInterpolator([0, 1, 2, 3], [2, 1, 4, 5], extrapolate=False)
func = PchipInterpolatorClip([0, 1, 2, 3], [2, 1, 4, 5])

x = np.linspace(-1, 4, 100)
# y = func(x)
# y = func(np.clip(x, 0, 3))
y = func(x)

plt.plot(x,y)

setFigPos("x: 201	y: 75	w: 3625	h: 2070")
plt.get_current_fig_manager().window.showMaximized()
fig.canvas.mpl_connect("key_press_event", close_figure)
fig.canvas.toolbar.zoom()
plt.show()
