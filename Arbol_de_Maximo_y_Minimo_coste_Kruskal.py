# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 01:01:58 2024

@author: moise
"""

import networkx as nx
import matplotlib.pyplot as plt

# Clase para representar el conjunto disjunto (disjoint set) para manejar la detección de ciclos
class DisjointSet:
    def __init__(self, vertices):
        self.parent = {v: v for v in vertices}
        self.rank = {v: 0 for v in vertices}

    def find(self, v):
        if self.parent[v] != v:
            self.parent[v] = self.find(self.parent[v])
        return self.parent[v]

    def union(self, v1, v2):
        root1 = self.find(v1)
        root2 = self.find(v2)
        if root1 != root2:
            if self.rank[root1] > self.rank[root2]:
                self.parent[root2] = root1
            elif self.rank[root1] < self.rank[root2]:
                self.parent[root1] = root2
            else:
                self.parent[root2] = root1
                self.rank[root1] += 1

def kruskal_mst(graph):
    mst = nx.Graph()  # Grafo para el MST resultante
    edges = list(graph.edges(data=True))
    edges.sort(key=lambda x: x[2]['weight'])  # Ordenar las aristas por peso
    ds = DisjointSet(graph.nodes)

    for edge in edges:
        node1, node2, weight = edge[0], edge[1], edge[2]['weight']
        if ds.find(node1) != ds.find(node2):
            mst.add_edge(node1, node2, weight=weight)
            ds.union(node1, node2)
            print(f"Arista añadida al MST: ({node1}, {node2}), peso: {weight}")

    return mst

# Crear un grafo de ejemplo
G = nx.Graph()
G.add_edge('A', 'B', weight=4)
G.add_edge('A', 'H', weight=8)
G.add_edge('B', 'H', weight=11)
G.add_edge('B', 'C', weight=8)
G.add_edge('H', 'I', weight=7)
G.add_edge('H', 'G', weight=1)
G.add_edge('I', 'C', weight=2)
G.add_edge('C', 'D', weight=7)
G.add_edge('C', 'F', weight=4)
G.add_edge('D', 'F', weight=14)
G.add_edge('D', 'E', weight=9)
G.add_edge('E', 'F', weight=10)
G.add_edge('F', 'G', weight=2)
G.add_edge('G', 'I', weight=6)

# Aplicar el algoritmo de Kruskal
mst = kruskal_mst(G)

# Dibujar el grafo original
plt.figure(figsize=(12, 6))

plt.subplot(121)
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=700, font_size=16)
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.title('Grafo Original')

# Dibujar el MST
plt.subplot(122)
pos = nx.spring_layout(mst)
nx.draw(mst, pos, with_labels=True, node_color='lightgreen', node_size=700, font_size=16)
labels = nx.get_edge_attributes(mst, 'weight')
nx.draw_networkx_edge_labels(mst, pos, edge_labels=labels)
plt.title('Árbol de Expansión Mínimo (MST) - Kruskal')

plt.show()

# Mostrar el progreso en la consola
print("Proceso del algoritmo de Kruskal completado.")
print("Nodos y aristas en el MST resultante:")
for edge in mst.edges(data=True):
    print(edge)
