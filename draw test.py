import json
import numpy as np

from Graph import Graph

nazwy = [
	"Grötzsch.json"    , # 0
	"Niespójny.json"   , # 1
	"Mchtr parter.json", # 2
	"Ość 9.json"       , # 3
	"Siatka 7x7.json"  , # 4
	"Losowy 2.json"    , # 5
	"W9.json"          , # 6
	"K33.json"         , # 7
	"Pettersen.json"   , # 8
	"K5.json"          , # 9
	"Losowy 1.json"    , # 10
]

with open("Przykładowe grafy/" + nazwy[8]) as f:
	d = json.load(f)
	# print(d)

graph = Graph((np.array(d["nodes"])-1).tolist())
# graph.generateRandomGraph(20, maxConnections=10)

# print(np.array(d["edges"]))
# print(np.array(d["edges"])-1)

# print((np.array(d["nodes"])-1).tolist())

# print(np.array(d["x"]))

graph.draw(x0=np.array(d["x"], dtype="float64").transpose(), simulate=False)
