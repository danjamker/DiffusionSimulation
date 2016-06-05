"""
Calculates each node's constraint value (structural holes) as described by Ronald Burt
"""

__author__ = """Alex Levenson (alex@isnontinvain.com)"""

#	(C) Reya Group: http://www.reyagroup.com
#	Alex Levenson (alex@isnotinvain.com)
#	BSD license.

import networkx as nx
import numpy


def _calcProportionaTieStrengthWRT(A, i):
    """
    Calculates P from Burt's equation
    using only the intersection of each node's ego network and that of node i

    A is the adjacencey matrix
    """
    num = A.copy()
    num = num + num.transpose()

    P = A.copy()
    mask = numpy.where(A, 1, 0)

    mask = mask[i]
    mask[0, i] = 1.0
    P = numpy.multiply(mask, P)
    P = numpy.multiply(mask.transpose(), P)
    P = P + P.transpose()

    denom = P.sum(1)
    denom = numpy.repeat(denom, len(P), axis=1)

    mask = numpy.where(denom, 1, float('nan'))
    denom = numpy.multiply(denom, mask)

    return numpy.nan_to_num(numpy.divide(num, denom))


def _calcProportionalTieStrengths(A):
    """
    Calculates P from Burt's equation,
    using each node's entire ego network

    A is the adjacencey matrix
    """
    num = A.copy()
    num = num + num.transpose()

    denom = num.sum(1)
    denom = numpy.repeat(denom, len(num), axis=1)
    mask = numpy.where(denom, 1, float('nan'))
    denom = numpy.multiply(denom, mask)
    return numpy.nan_to_num(numpy.divide(num, denom))


def _neighborsIndexes(graph, node, includeOutLinks, includeInLinks):
    """
    returns the neighbors of node in graph
    as a list of their INDEXes within graph.node()
    """
    neighbors = set()

    if includeOutLinks:
        neighbors |= set(graph.neighbors(node))

    if includeInLinks:
        neighbors |= set(graph.predecessors(node))

    return map(lambda x: graph.nodes().index(x), neighbors)


def structural_holes(graph, includeOutLinks=True, includeInLinks=False, wholeNetwork=True):
    """
    Calculate each node's contraint / structural holes value, as described by Ronal Burt

    Parameters
    ----------
    G : graph
        a networkx Graph or DiGraph

    includeInLinks : whether each ego network should include nodes which point to the ego - this should be False for undirected graphs

    includeOutLinks : whether each ego network should include nodes which the ego points to - this should be True for undirected graphs

    wholeNetwork : whether to use the whole ego network for each node, or only the overlap between the current ego's network and the other's ego network

    Returns
    -------
    constraints : dictionary
                  dictionary with nodes as keys and dictionaries as values in the form {"C-Index": v,"C-Size": v,"C-Density": v,"C-Hierarchy": v}
                  where v is each value

    References
    ----------
    .. [1] Burt, R.S. (2004). Structural holes and good ideas. American Journal of Sociology 110, 349-399
    """

    if not hasattr(graph, "predecessors"):
        print "graph is undirected... setting includeOutLinks to True and includeInLinks to False"
        includeOutLinks = True
        includeInLinks = False

    # get the adjacency matrix view of the graph
    # which is a numpy matrix
    A = nx.to_numpy_matrix(graph)

    # calculate P_i_j from Burt's equation
    p = _calcProportionalTieStrengths(A)

    # this is the return value
    constraints = {}

    for node in graph.nodes():
        # each element of constraints will be a dictionary of this form
        # unless the node in question is an isolate in which case it
        # will be None
        constraint = {"C-Index": 0.0, "C-Size": 0.0, "C-Density": 0.0, "C-Hierarchy": 0.0}

        # Vi is the set of i's neighbors
        Vi = _neighborsIndexes(graph, node, includeOutLinks, includeInLinks)
        if len(Vi) == 0:
            # isolates have no defined constraint
            constraints[node] = None
            continue

        # i is the node we are calculating constraint for
        # and is thus the ego of the ego net
        i = graph.nodes().index(node)

        if not wholeNetwork:
            # need to recalculate p w/r/t this node
            pq = _calcProportionaTieStrengthWRT(A, i)
        else:
            # don't need to calculate p w/r/t any node
            pq = p

        for j in Vi:
            Pij = p[i, j]
            constraint["C-Size"] += Pij ** 2
            innerSum = 0.0
            for q in Vi:
                if q == j or q == i: continue
                innerSum += p[i, q] * pq[q, j]

            constraint["C-Hierarchy"] += innerSum ** 2
            constraint["C-Density"] += 2 * Pij * innerSum

        constraint["C-Index"] = constraint["C-Size"] + constraint["C-Density"] + constraint["C-Hierarchy"]
        constraints[node] = constraint
    return constraints