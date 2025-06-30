import random
from Graph import Graph
# import sys
# sys.setrecursionlimit(1500)

graph = Graph([[0]])
graph.generateRandomGraph(20, maxConnections=(2,6))

graph.print()

path = graph.findPath(0, 6, Graph.PathFindingAlgorithms.DEPTH_FIRST_SEARCH)
print(path)

Graph.simpleTraversePath(graph, path)
