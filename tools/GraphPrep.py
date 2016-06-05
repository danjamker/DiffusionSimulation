import logging

import community
import networkx as nx
import pandas as pd

import EdgeSwapGraph


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
        logging.info('Starting Edge Swap')

        self.G = nx.double_edge_swap(self.G, 50, 150)
        logging.info('Finished Edge Swap')

        return self

    def detect_communities(self):
        logging.info('Starting Community Detection')

        partition = community.best_partition(self.G)
        for n in partition:
            nx.set_node_attributes(self.G, 'community', {n: partition[n]})

        logging.info('Finished Community Detection')

        self.l.append("community")
        return self

    def save_to_pickle(self, path):
        logging.info('Saving to pickle')
        nx.write_gpickle(self.G, path, protocol=2)
        logging.info('Saved to pickle')

        return self

    def save_edges_to_csv(self, path):
        logging.info('Saving edges to CSV')

        fh = open(path, 'wb')
        nx.write_edgelist(self.G, fh, delimiter=',', data=self.l)
        logging.info('Saved  edges to csv')
        return self

    def save_nodes_to_csv(self, path):
        logging.info('Saving nodes to CSV')

        fh = open(path, 'wb')
        for n in self.G.nodes(data=True):
            line = []
            line.append(str(n[0]))
            for x in self.l:
                if x in n[1]:
                    line.append(str(n[1][x]))
            fh.write(",".join(line))
            fh.write("\n")
        logging.info('Saved nodes to csv')

        return self

    def to_undriected(self):
        logging.info('Conversting to undirected graph')

        G_tmp = self.G.to_undirected()  # copy
        for (u, v) in self.G.edges():
            if not self.G.has_edge(v, u):
                G_tmp.remove_edge(u, v)
            else:
                avr = (self.G.get_edge_data(u, v)["weight"] + self.G.get_edge_data(v, u)["weight"]) / 2
                G_tmp.add_edge(u, v, weight=avr)
        self.G = G_tmp
        logging.info('Converted to undirected graph')

        return self

    def remove_self_loop(self):
        logging.info('Removing self loops')
        self.G.remove_edges_from(self.G.selfloop_edges())
        logging.info('Removed self loops')
        return self

    def remove_unconnected_nodes(self):
        logging.info('Removing unconnected nodes')

        outdeg = self.G.degree()
        to_remove = [n for n in outdeg if outdeg[n] == 0]
        self.G.remove_nodes_from(to_remove)
        logging.info('Removed unconnected nodes')

        return self

    def edge_swap(self, iterations=10):
        logging.info('Swapping edges')

        EdgeSwapGraph.randomize_by_edge_swaps(iterations, self.G)
        logging.info('Swapped edges')

        return self

    def pagerank(self, alpha=0.85):
        logging.info('Applying pagerank')
        pr = nx.pagerank(self.G, alpha=alpha)
        for n in pr:
            nx.set_node_attributes(self.G, 'pagerank', {n: pr[n]})
        self.l.append("pagerank")
        logging.info('Applied pagerank')

        return self

    def add_node_atributes(self, path):
        df = pd.read_csv(path).set_index("node")
        for c in df.columns.values.tolist():
            self.l.append(c)
        for index, row in df[(df.index.isin(self.G.nodes()))].iterrows():
            for c in df.columns.values.tolist():
                nx.set_node_attributes(self.G, c, {index: row[c]})
        return self

    def save_to_gml(self, path):
        nx.write_gml(self.G, path)
        return self

    def save_to_graphml(self, path):
        nx.write_graphml(self.G, path)
        return self
if __name__ == '__main__':
    # GraphPrep(
    #     "/Users/kershad1/Downloads/twitter_geo_network"). \
    #     edge_swap(iterations=10). \
    #     save_edges_to_csv("./../data/shuffel/twitter_geo_network_shuffel_2.csv")
    #
    # GraphPrep(
    #     "/Users/kershad1/Downloads/twitter_mention_network"). \
    #     edge_swap(iterations=10). \
    #     save_edges_to_csv("./../data/shuffel/twitter_mention_network_shuffel_2.csv")
    #

    # GraphPrep(
    #     "/Users/kershad1/Downloads/reddit_comment_network"). \
    #     edge_swap(iterations=10). \
    #     save_edges_to_csv("./../data/shuffel/reddit_comment_network_shuffel_2.csv")
    #
    # GraphPrep(
    #     "/Users/kershad1/Downloads/reddit_traversal_network"). \
    #     edge_swap(iterations=10).save_edges_to_csv("./../data/shuffel/reddit_traversal_network_shuffel_3.csv")

    GraphPrep(
        "/Users/danielkershaw/Downloads/twitter_geo_network.csv"). \
        remove_unconnected_nodes(). \
        remove_self_loop(). \
        pagerank(). \
        save_to_graphml("../networks/twitter_geo_network_pagerank_directed.graphml")

    # GraphPrep(
    #     "/Users/kershad1/Downloads/reddit_traversal_network"). \
    #     to_undriected(). \
    #     remove_unconnected_nodes(). \
    #     remove_self_loop(). \
    #     detect_communities(). \
    #     pagerank(). \
    #     save_to_pickle("../data/pickle/reddit_traversal_network_pagerank.gpickle")
    #
    # GraphPrep(
    #     "/Users/kershad1/Downloads/twitter_mention_network"). \
    #     to_undriected(). \
    #     remove_unconnected_nodes(). \
    #     remove_self_loop(). \
    #     detect_communities(). \
    #     pagerank(). \
    #     save_to_pickle("../data/pickle/twitter_mention_network_pagerank.gpickle")
    #
    # GraphPrep(
    #     "/Users/kershad1/Downloads/reddit_comment_network"). \
    #     to_undriected(). \
    #     remove_unconnected_nodes(). \
    #     remove_self_loop(). \
    #     detect_communities(). \
    #     pagerank(). \
    #     save_to_pickle("../data/pickle/reddit_comment_network_pagerank.gpickle")
