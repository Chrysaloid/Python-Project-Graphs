import random

# from codetiming import Timer

from utils import NestedExit

random.seed(0)

class	Graph:
	def __init__(self, graph: list | int = []):
		self.graph = []

		self._visited = []

		if isinstance(graph, int):
			self.generateRandomGraph(graph)
		elif not graph:
			self.generateRandomGraph()
		elif isinstance(graph, list):
			# self.setGraph(graph)
			self.graph = graph
		else:
			raise TypeError("graph should be either list or int")

	# def setGraph(self, graph):
	# 	self.graph = graph

	# def getGraph(self):
	# 	return self.graph

	@property
	def len(self):
		return len(self.graph)

	def _initVisited(self, val = False):
		# self._visited = [val] * self.len
		if len(self._visited) == self.len:
			for i in range(self.len):
				self._visited[i] = val
		else:
			self._visited = [val] * self.len

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

	def generateRandomGraph(
			self,
			numNodes = 10,
			makeConnected = True,
			maxConnections: int | tuple[int, int] = None,
			multiEdge = False,
			selfLoop = False,
			directional = False,
		):
		if not directional:
			selfLoop = False
		if maxConnections is None:
			minConn = 0
			maxConn = numNodes
		elif isinstance(maxConnections, int):
			minConn = 0
			maxConn = min(maxConnections, numNodes) if not multiEdge else maxConnections
		else:
			minConn = max(maxConnections(0), 0)
			maxConn = min(maxConnections(1), numNodes) if not multiEdge else maxConnections
		idxes = range(numNodes)
		while True:
			if multiEdge:
				self.graph = [random.choices(idxes, k=random.randrange(minConn, maxConn)) for i in idxes]
			else:
				self.graph = [set(random.sample(idxes, random.randrange(minConn, maxConn))) for i in idxes]
			if not makeConnected or self.isConnected():
				break
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

	# @Timer(text = lambda sec: f"{sec * 1000:.1f} ms")
	def findPath(self, start: int, end: int):
		if not isinstance(start, int):
			raise TypeError("start should be an int.")
		if not 0 <= start <= len(self.graph):
			raise ValueError("start should be between 0 and len(self.graph).")
		if not isinstance(end, int):
			raise TypeError("end should be an int.")
		if not 0 <= end <= len(self.graph):
			raise ValueError("end should be between 0 and len(self.graph).")

		self._initVisited()

		self._path = []
		try:
			self._findPath(start, end)
		except NestedExit:
			pass

		# self.path = [start] + self.__findPath0(start, end) ### Slower

		return self._path

	def _findPath(self, current, end):
		self._visited[current] = True
		self._path.append(current)
		for node in self.graph[current]:
			if self._visited[node]:
				continue
			if node == end:
				self._path.append(end)
				raise NestedExit
			else:
				self._findPath(node, end)
		self._path.pop()

	# def __findPath0(self, current, end): ### Slower
	# 	self.visited[current] = True
	# 	for node in self.graph[current]:
	# 		if self.visited[node]:
	# 			continue
	# 		if node == end:
	# 			return [end]
	# 		else:
	# 			temp = self.__findPath0(node, end)
	# 			if temp:
	# 				return [node] + temp
	# 	return False

