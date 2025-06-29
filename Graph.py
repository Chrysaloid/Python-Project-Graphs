import random
import math
from collections import deque

# from codetiming import Timer

from utils import NestedExit
from PathFindingAlgorithms import PathFindingAlgorithms as PathAlgs

# random.seed(0)

class	Graph:
	PathFindingAlgorithms = PathAlgs

	def __init__(self, graph: list | int = []):
		self.graph = []

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
	def simpleTraversePath(graph: "Graph", path: list[int]):
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
		if not self.graph:
			return True

		if self.directional:
			for neighbourhood in self.graph: # check for isolated nodes == with no neighbours
				if not neighbourhood:
					return False

		self._initVisited()

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

	# @Timer(text = lambda sec: f"{sec * 1000:.1f} ms")
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

				# self.path = [start] + self.__findPath0(start, end) ### Slower

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

