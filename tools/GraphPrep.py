from random import shuffle

import community
import networkx as nx
import pandas as pd


class GraphPrep:
    def __init__(self, path):
        '''Constructor'''
        edges = pd.read_csv(path, header=0)
        self.l = edges.columns.values.tolist()
        self.l.remove('source')
        self.l.remove('target')
        self.G = nx.from_pandas_dataframe(edges, 'source', 'target', self.l, create_using=nx.DiGraph())

    def swap_edges(self):
        # Random small world
        # Random mix of edges
        # Reverso of edges
        self.G = nx.double_edge_swap(self.G, 50, 150)
        return self

    def shuffle_target(self):
        edges = self.G.edges(data=True)

        for e in edges:
            self.G.remove_edge(e[0], e[1])

        target = [e[1] for e in edges]
        shuffle(target)

        for e1, e2 in zip(edges, target):
            self.G.add_edge(e1[0], e2, e1[2])

        edges = self.G.edges(data=True)

        return self

    def shuffle_source(self):
        edges = self.G.edges(data=True)

        for e in edges:
            self.G.remove_edge(e[0], e[1])

        target = [e[0] for e in edges]
        shuffle(target)

        for e1, e2 in zip(edges, target):
            self.G.add_edge(e2, e1[1], e1[2])

        return self

    def detect_communities(self):
        partition = community.best_partition(self.G)
        for n in partition:
            nx.set_node_attributes(self.G, 'community', {n: partition[n]})

        self.l.append("community")
        return self

    def save_to_pickle(self, path):
        nx.write_gpickle(self.G, path, protocol=2)
        return self

    def save_edges_to_csv(self, path):
        fh = open(path, 'wb')
        nx.write_edgelist(self.G, fh, delimiter=',', data=self.l)
        return self

    def save_nodes_to_csv(self, path):
        fh = open(path, 'wb')
        for n in self.G.nodes(data=True):
            line = []
            line.append(str(n[0]))
            for x in self.l:
                if x in n[1]:
                    line.append(str(n[1][x]))
            fh.write(",".join(line))
            fh.write("\n")
        return self

    def deg_graph(self):
        return self.G

    def to_undriected(self):
        G_tmp = self.G.to_undirected()  # copy
        for (u, v) in self.G.edges():
            if not self.G.has_edge(v, u):
                G_tmp.remove_edge(u, v)
            else:
                avr = (self.G.get_edge_data(u, v)["weight"] + self.G.get_edge_data(v, u)["weight"]) / 2
                G_tmp.add_edge(u, v, weight=avr)
        self.G = G_tmp
        return self

    def normalise(self):

        return self

if __name__ == '__main__':
    GraphPrep(
        "/Users/danielkershaw/PycharmProjects/DiffusionSimulation/data/Twitter_GEO_network.csv").to_undriected().detect_communities().save_to_pickle(
        "./../data/go-pickle.gpickle")
    # save_to_pickle("../data/Twitter_GEO_network.pickle")
