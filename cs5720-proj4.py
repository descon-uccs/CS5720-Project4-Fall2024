# Fixing the Python code to use hyphens consistently in graph type names for filenames

import os
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from heapq import heappop, heappush

# Load graphs and perform basic analysis
def count_nodes_and_edges(file_path):
    graph_matrix = pd.read_csv(file_path, header=None)
    num_nodes = graph_matrix.shape[0]
    num_edges = (graph_matrix != -1).sum().sum() // 2  # Undirected graph
    return num_nodes, num_edges

def classify_graph_type(edges, nodes):
    density = edges / (nodes * (nodes - 1) / 2)
    return "Sparse" if density <= 0.5 else "Dense"

# Prim's Algorithm (Adjacency Matrix + Unordered Array Priority Queue)
def prim_mst(matrix):
    num_nodes = len(matrix)
    visited = [False] * num_nodes
    total_weight = 0
    edges = [(0, 0)]
    while edges:
        weight, node = heappop(edges)
        if visited[node]:
            continue
        total_weight += weight
        visited[node] = True
        for neighbor in range(num_nodes):
            if not visited[neighbor] and matrix[node][neighbor] != -1:
                heappush(edges, (matrix[node][neighbor], neighbor))
    return total_weight

# Kruskal's Algorithm
def kruskal_mst(matrix):
    num_nodes = len(matrix)
    edges = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if matrix[i][j] != -1:
                edges.append((matrix[i][j], i, j))
    edges.sort()
    parent = list(range(num_nodes))

    def find(x):
        while x != parent[x]:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        root_x, root_y = find(x), find(y)
        if root_x != root_y:
            parent[root_y] = root_x

    total_weight = 0
    for weight, u, v in edges:
        if find(u) != find(v):
            union(u, v)
            total_weight += weight
    return total_weight

# Timing Analysis
def timing_analysis(func, matrix):
    start = time.time()
    func(matrix)
    return time.time() - start

# Paths and Setup
data_path = "graphs"
results = {"Graph": [], "Nodes": [], "Edges": [], "Density": [], "Prim MST": [], "Kruskal MST": [], "Prim Time": [], "Kruskal Time": []}

for graph_type in ["type-1", "type-2", "type-3"]:
    type_path = os.path.join(data_path, graph_type.replace("-", "_"))
    for file_name in os.listdir(type_path):
        graph_path = os.path.join(type_path, file_name)
        graph_matrix = pd.read_csv(graph_path, header=None).to_numpy()
        
        nodes, edges = count_nodes_and_edges(graph_path)
        density = classify_graph_type(edges, nodes)
        
        prim_weight = prim_mst(graph_matrix)
        kruskal_weight = kruskal_mst(graph_matrix)
        prim_time = timing_analysis(prim_mst, graph_matrix)
        kruskal_time = timing_analysis(kruskal_mst, graph_matrix)
        
        results["Graph"].append(f"{graph_type} - {file_name}")
        results["Nodes"].append(nodes)
        results["Edges"].append(edges)
        results["Density"].append(density)
        results["Prim MST"].append(prim_weight)
        results["Kruskal MST"].append(kruskal_weight)
        results["Prim Time"].append(prim_time)
        results["Kruskal Time"].append(kruskal_time)

# Save Results
results_df = pd.DataFrame(results)
results_df.to_csv("graph-analysis-results.csv", index=False)

# Plotting
for graph_type in ["type-1", "type-2", "type-3"]:
    subset = results_df[results_df["Graph"].str.contains(graph_type)]
    plt.figure()
    plt.title(f"MST Weights: {graph_type}")
    plt.plot(subset["Graph"], subset["Prim MST"], label="Prim")
    plt.plot(subset["Graph"], subset["Kruskal MST"], label="Kruskal")
    plt.xlabel("Graph")
    plt.ylabel("MST Weight")
    plt.legend()
    plt.xticks(rotation=45, fontsize=8)
    plt.tight_layout()
    plt.savefig(f"mst-weights-{graph_type}.png")

    plt.figure()
    plt.title(f"Timing Analysis: {graph_type}")
    plt.plot(subset["Graph"], subset["Prim Time"], label="Prim")
    plt.plot(subset["Graph"], subset["Kruskal Time"], label="Kruskal")
    plt.xlabel("Graph")
    plt.ylabel("Time (s)")
    plt.legend()
    plt.xticks(rotation=45, fontsize=8)
    plt.tight_layout()
    plt.savefig(f"timing-analysis-{graph_type}.png")
