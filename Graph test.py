import timeit
from statistics import fmean, stdev

from Graph import Graph


graph = Graph([])
graph.generateRandomGraph(200, maxConnections=10)

# for s in graph.getGraph():
# 	print(s)
def test(): return graph.findPath(0, 29)
print(test())
print(len(test()) == len(set(test())))

number = 2000
times = list(map(lambda x: x * 1e6 / number, timeit.repeat(stmt = test, repeat=10, number=number)))
print(f"{fmean(times):.1f} us ± {stdev(times) / fmean(times) * 100:.1f} %")
