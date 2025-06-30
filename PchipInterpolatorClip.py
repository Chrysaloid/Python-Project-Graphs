import numpy as np
from scipy.interpolate import PchipInterpolator
class PchipInterpolatorClip(PchipInterpolator):
	def __init__(self, x, y):
		super().__init__(x, y, extrapolate=False)

	def __call__(self, xq):
		return super().__call__(np.clip(xq, self.x[0], self.x[-1]))
