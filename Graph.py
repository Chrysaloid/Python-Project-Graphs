import random
import math
import numpy as np
from collections import deque
from matplotlib import pyplot as plt
from time import time
from scipy.interpolate import interp1d
from collections.abc import Callable

from utils import NestedExit, close_figure, setFigPos, transformToUnitSquarePadded
from PchipInterpolatorClip import PchipInterpolatorClip
from PathFindingAlgorithms import PathFindingAlgorithms as PathAlgs

plt.style.use("dark_background")

class	Graph:
	def __init__(self, graph: list | int = []):
		self.graph = []
		# self.edges = []

		self._visited = []

		self.multiEdge   = False
		self.selfLoop    = False
		self.directional = False

		if isinstance(graph, int):
			self.generateRandomGraph(graph)
		elif not graph:
			self.generateRandomGraph()
		elif isinstance(graph, list):
			# self.setGraph(graph)
			self.graph = graph
		else:
			raise TypeError("graph should be either list or int")

	@property
	def len(self):
		return len(self.graph)

	@staticmethod
	def simpleTraversePath(graph: "Graph", path: list[int]) -> bool:
		g = graph.graph
		for i in range(len(path)-1):
			if path[i+1] not in g[path[i]]:
				print("Invalid path")
				return False
		print("Path OK")
		return True

	def print(self):
		print("[")
		field_size = math.floor(math.log10(self.len - 1)) + 1
		for i in range(self.len):
			print(" {i:{field_size}}: {neighbourhood}".format(i=i, field_size=field_size, neighbourhood=" ".join(map(str, self.graph[i]))))
		print("]")

	def _initVisited(self, val = False):
		# self._visited = [val] * self.len
		if len(self._visited) == self.len:
			for i in range(self.len):
				self._visited[i] = val
		else:
			self._visited = [val] * self.len

	def isConnected(self):
		""" Breadth-first search method to check if graph is connected """
		if not self.graph:
			return False

		if self.directional:
			for neighbourhood in self.graph: # check for isolated nodes == with no neighbours
				if not neighbourhood:
					return False

		self._initVisited()

		"""
		Trudno tutaj dokonać abstrakcji jedynie samego algorytmu i wiązałoby się to z utratą wydajności.
		Jednak abstrakcja teoretycznie jest możliwa.
		Należałoby powsawdzać gdzie się da wyrażenia lambda które byłyby podawane do funkcji jako parametry.
		Te lambdy decydowałyby np. o momentach wczesnego wyjścia z algorytmu i co się dzieje przed i po pętli for.
		Lambdy otrzymywałyby zmienne takie jak `current` czy `neighbour` jako parametry a inne przez *clousure*.
		Domyślną wartością lambd-parametrów byłoby `None` i każde ich wywołanie byłoby poprzedzone `if fun is not None:`.
		Jeżeli potraktować taką ogólną funkcję jako bazę do bardziej wyspecjalizowanych (mającyh wypełnionych tylko kilka lambd)
		to może kompilator byłby w stanie zoptymalizować te zewnętrzne funkcje poprzez usunięcie tych wywołań i ich sprawdzeń które są niezdefiniowane.
		"""
		queue = deque([0])
		while len(queue):
			current = queue.popleft()
			for neighbour in self.graph[current]:
				if self._visited[neighbour]:
					continue
				queue.append(neighbour)
				self._visited[neighbour] = True

		return all(self._visited)

	"""
	def isConnected(self):
		if not self.graph:
			return True

		self._initVisited()
		# start = random.randrange(self.len)
		start = 0
		self._isConnected(start)

		return all(self._visited)

	def _isConnected(self, current):
		self._visited[current] = True
		for node in self.graph[current]:
			if self._visited[node]:
				continue
			else:
				self._isConnected(node)
	"""

	def generateRandomGraph(
			self,
			numNodes = 10,
			makeConnected = True,
			maxConnections: int | tuple[int, int] = None,
			multiEdge = False,
			selfLoop = False,
			directional = False,
		):
		self.multiEdge   = multiEdge
		self.selfLoop    = selfLoop
		self.directional = directional

		minConn = 1 if makeConnected else 0
		if maxConnections is None:
			maxConn = numNodes
		elif isinstance(maxConnections, int):
			maxConn = min(maxConnections, numNodes) if not multiEdge else maxConnections
		else:
			minConn = max(maxConnections[0], minConn)
			maxConn = min(maxConnections[1], numNodes) if not multiEdge else maxConnections[1]

		idxes = range(numNodes)
		# attempts = 0
		while True:
			if multiEdge: # k is the node neighbourhood size
				self.graph = [random.choices(idxes, k=random.randrange(minConn, maxConn)) for i in idxes]
			else:
				self.graph = [set(random.sample(idxes, k=random.randrange(minConn, maxConn))) for i in idxes]
			# attempts += 1
			if not makeConnected or self.isConnected():
				break
		# print(attempts)

		if not selfLoop:
			listSet = list if multiEdge else set
			for i in idxes:
				self.graph[i] = listSet(filter(lambda x: x != i, self.graph[i]))

		if not directional:
			if multiEdge:
				temp = [[]] * numNodes
				for i in idxes:
					for node in self.graph[i]:
						temp[node].append(i)
				for i in idxes:
					self.graph[i] += temp[i]
			else:
				for i in idxes:
					for node in self.graph[i]:
						self.graph[node].add(i)

	PathFindingAlgorithms = PathAlgs
	def findPath(
			self,
			start: int,
			end: int,
			algorithm: PathAlgs = PathAlgs.DEPTH_FIRST_SEARCH
		) -> list[int]:
		"""
		Finds path between @start and @end nodes using specified algorithm.
		If path is not found - empty list is returned.

		args:
			start (int): Start node
			end (int): End node

		return:
			Path as list of node indices.
		"""
		if not isinstance(start, int):
			raise TypeError("start should be an int.")
		if not 0 <= start < self.len:
			raise ValueError("start should be between 0 and graph.len exlusive.")
		if not isinstance(end, int):
			raise TypeError("end should be an int.")
		if not 0 <= end < self.len:
			raise ValueError("end should be between 0 and graph.len exlusive.")

		if start == end:
			if self.selfLoop and start in self.graph[start]:
				self._path = [start, start]
			else:
				self._path = []
			return self._path

		match algorithm:
			case PathAlgs.DEPTH_FIRST_SEARCH:
				self._initVisited()
				self._path = []

				try:
					self._findPath_DEPTH_FIRST_SEARCH(start, end)
				except NestedExit:
					pass

				return self._path
			case PathAlgs.BREADTH_FIRST_SEARCH:
				self._initVisited()

				queue = deque([start])
				enterNode = [None] * self.len
				while len(queue):
					current = queue.popleft()
					for neighbour in self.graph[current]:
						if self._visited[neighbour]:
							continue
						enterNode[neighbour] = current
						if neighbour == end:
							break
						queue.append(neighbour)
						self._visited[neighbour] = True
					else: # break did not happened -> end not found -> do not break while loop
						continue
					break
				else: # path was not found
					self._path = []
					return self._path

				current = end
				self._path = deque([])
				while current != start:
					self._path.appendleft(current)
					current = enterNode[current]
				self._path.appendleft(start)

				self._path = list(self._path)
				return self._path

	def _findPath_DEPTH_FIRST_SEARCH(self, current, end):
		""" Helper recursive function """
		self._visited[current] = True
		self._path.append(current)
		for node in self.graph[current]: # loop over connected nodes
			if self._visited[node]:
				continue
			if node == end:
				self._path.append(end)
				raise NestedExit
			else:
				self._findPath_DEPTH_FIRST_SEARCH(node, end)
		self._path.pop() # path was not found so to return empty list we pop the only inserted element

	matlabBlue = (0.000,0.447,0.741)
	FcFunBase = PchipInterpolatorClip([0, 1, 2, 3], [2, 1, 4, 5])
	FsFunBase = PchipInterpolatorClip([0, 1, 2, 3], [2, 1, 4, 5])
	FfFunBase = interp1d([0, 1, 2, 3], [2, 1, 4, 5], kind="linear", fill_value="extrapolate", copy=False)
	def draw(
		self,
		simulate: bool = True, # distribute nodes with a simulation
		q: float = 1.0, # Coulomb's coefficient
		k: float = 3.0, # coefficient of spring force
		u: float = -1.0, # coefficient of friction force
		m: float = 1.0, # node mass
		dt: float = 1e-2, # time delta of simulation
		FcFun: Callable = FcFunBase, # Coulomb's force
		FsFun: Callable = FsFunBase, # spring force
		FfFun: Callable = FfFunBase, # friction force
		maxSpeedThreshold: float = 0.001,
		maxI: float = 1e4,
		logSpeed: bool = True,
		logSpeedInter: float = 0.1, # s
		unitSquarePerc: float = 0.9,
		animate: bool = True,
		x0: np.ndarray = None, # initial positions
	):
		if not self.graph:
			raise RuntimeWarning("Can't draw an empty graph")

		edges = set()
		for i in range(self.len):
			for node in self.graph[i]:
				if i != node:
					edges.add(frozenset((i,node)))
		# edges = list(edges)
		edges = np.array([list(x) for x in edges])
		# print(edges)

		fig, ax = plt.subplots(tight_layout=True)
		# plt.xticks([]), plt.yticks([])
		ax.set_aspect("equal", adjustable="box")
		setFigPos("x: 201	y: 75	w: 3625	h: 2070")
		plt.get_current_fig_manager().window.showMaximized()
		fig.canvas.mpl_connect("key_press_event", close_figure)
		fig.canvas.toolbar.zoom()

		z0 = np.zeros((2,self.len))
		xdata = np.stack((z0[0,edges[:,0]], z0[0,edges[:,1]]), axis=0)
		ydata = np.stack((z0[1,edges[:,0]], z0[1,edges[:,1]]), axis=0)
		p1 = plt.plot(xdata,ydata, color=self.matlabBlue)
		z1 = np.zeros(self.len)
		p2 = plt.plot(z1, z1, color=self.matlabBlue, linestyle="none", marker=".", markersize=12)[0]

		plt.show(block=False)

		if x0 is None:
			x0 = np.random.rand((2, self.len)) * 0.4
		elif type(x0) is not np.ndarray:
			raise ValueError("x0 should be np.ndarray")
		elif x0.ndim != 2:
			raise ValueError("x0 should be 2D")
		elif x0.shape[0] != 2 or x0.shape[1] != self.len:
			raise ValueError("x0's shape should be 2 x self.len")
		else:
			transformToUnitSquarePadded(x0, unitSquarePerc)

		def draw_helper(x):
			xdata = np.stack((x[0,edges[:,0]], x[0,edges[:,1]]), axis=0)
			ydata = np.stack((x[1,edges[:,0]], x[1,edges[:,1]]), axis=0)
			for j in range(edges.shape[0]):
				p1[j].set_xdata(xdata[:,j])
				p1[j].set_ydata(ydata[:,j])
			p2.set_xdata(x[0,:])
			p2.set_ydata(x[1,:])
			ax.relim()
			ax.autoscale_view()
			fig.canvas.draw_idle()

		if simulate:
			x1 = x0.copy()
			x2 = x0.copy()

			poprzCzas = 0.0
			i = 0
			maxSpeed = math.inf
			t0 = time()
			while maxSpeed > maxSpeedThreshold and i < maxI:
				for j in range(self.len):
					currentPoint = x0[:,j]

					# Coulomb's force
					rj = x0 - currentPoint
					rjLen = np.linalg.norm(rj, axis=0)
					Fc = q*np.nansum((rj/rjLen) * FcFun(rjLen)) # q/r^2

					# spring force
					rl = x0[:, self.graph[j]] - currentPoint
					rlLen = np.linalg.norm(rl, axis=0)
					Fs = k*np.nansum((rl/rlLen) * FsFun(rlLen)) # k*(r - r0)

					# friction force
					v = x1[:,j] - currentPoint
					vLen = np.linalg.norm(v, axis=0)
					Ff = (v/vLen) * (u*FfFun(vLen/dt)) # u*V
					if Ff is np.nan: Ff = 0

					# discretized differential equation to calculate displacement from resultant force
					x2[:,j] = dt^2/m*(Fc + Fs + Ff) + 2*x1[:,j] - currentPoint

				if animate:
					draw_helper(x2)

				czas = time() - t0
				i += 1

				x0, x1, x2 = x1, x2, x0

				maxSpeed = max(np.linalg.norm(x1 - x0, axis=1)) / dt
				if logSpeed and czas - poprzCzas > logSpeedInter: # s
					print(f"{i:06d} | Max speed: {maxSpeed:.4f}")
					poprzCzas = czas

			print(f"{i:06d} | Max speed: {maxSpeed:.4f}")

			# x2 = transformToUnitSquarePadded(x2,unitSquarePerc)
		else:
			draw_helper(x0)

		plt.show(block=True)
