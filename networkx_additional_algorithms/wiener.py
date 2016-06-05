import networkx as nx

def wiener_index(self, G, weight=None):
    from itertools import chain

    is_directed = G.is_directed()
    if (is_directed and not nx.components.is_strongly_connected(G)) or \
            (not is_directed and not nx.components.is_connected(G)):
        return float('inf')
    pp = nx.shortest_paths.shortest_path_length(G, weight=weight)
    cc = [p.values() for k, p in pp.iteritems()]
    total = sum(chain.from_iterable(cc))
    # Need to account for double counting pairs of nodes in undirected graphs.
    return total if is_directed else total / 2