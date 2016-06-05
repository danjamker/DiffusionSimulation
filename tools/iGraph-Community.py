import louvain
import igraph as ig


if __name__ == '__main__':
    G = ig.Graph.Erdos_Renyi(100, 0.1);
    part = louvain.find_partition(G, method='Modularity');
