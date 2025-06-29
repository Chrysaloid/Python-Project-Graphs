import timeit
from statistics import fmean, stdev

from Graph import Graph

graph = Graph([])
graph.generateRandomGraph(20, maxConnections=10)

