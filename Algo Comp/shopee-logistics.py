import networkx as nx
from itertools import combinations 

G = nx.Graph()

n = eval(input())
for i in range(n - 1):
       n1, n2, l = input().split(' ')
       G.add_edge(n1, n2, length = int(l))

def path_length(source, target):
       paths = sorted(nx.all_simple_paths(G, source, target))
       return max([sum([G[path[i]][path[i + 1]]['length'] for i in range(len(path) - 1)]) for path in paths])         
       
print(sorted([path_length(pair[0], pair[1]) for pair in list(combinations(list(G.nodes), 2))])[-2])