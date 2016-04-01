import community
import networkx as nx
import pandas as pd


class GraphPrep:
    def __init__(self, path):
        '''Constructor'''
        edges = pd.read_csv(path, header=0)
        print(edges.columns.values)
        if "weight" in edges.columns.values.tolist():
            self.G = nx.from_pandas_dataframe(edges, 'source', 'target', ['weight'])
        else:
            self.G = nx.from_pandas_dataframe(edges, 'source', 'target')

    def swap_edges(self):
        # Random small world
        # Random mix of edges
        # Reverso of edges
        self.G = nx.double_edge_swap(self.G, 1, 100)
        return self

    def detect_communities(self):
        partition = community.best_partition(self.G)
        for n in partition:
            nx.set_node_attributes(self.G, 'community', {n: partition[n]})
        return self

    def save_to_pickle(self, path):
        nx.write_gpickle(self.G, path, protocol=2)
        return self

    def deg_graph(self):
        return self.G


if __name__ == '__main__':
    GraphPrep("/Users/danielkershaw/PycharmProjects/DiffusionSimulation/data/Twitter_GEO_network.csv"). \
        swap_edges(). \
        detect_communities(). \
        save_to_pickle("/Users/danielkershaw/PycharmProjects/DiffusionSimulation/data/Twitter_GEO_network.pickle")
