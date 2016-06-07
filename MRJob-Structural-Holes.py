from __future__ import division

try:
    from BytesIO import BytesIO
except ImportError:
    from io import BytesIO

try:
    from urlparse import urlparse
except ImportError:
    from urllib.parse import urlparse

from mrjob.job import MRJob
from mrjob.protocol import JSONValueProtocol
from mrjob.step import MRStep

from networkx_additional_algorithms import structural_holes
from networkx_additional_algorithms import brokerage
import networkx as nx
import pandas as pd
class MRJobStructuralHoles(MRJob):
    OUTPUT_PROTOCOL = JSONValueProtocol

    def configure_options(self):
        super(MRJobStructuralHoles, self).configure_options()

        self.add_file_option('--network')

        self.add_passthrough_option('--wholeNetwork', type='int', default=1, help='Should this be computed across the whole network <structural-holes>')
        self.add_passthrough_option('--messure', type='string', default="strutural-holes", help='which messur to run, brokerage or strutural-holes')
        self.add_passthrough_option('--includeOutLinks', type='int', default=1, help='for structural holes should outlinks be included')
        self.add_passthrough_option('--includeInLinks', type='int', default=0, help='for structural holes should outlinks be included')
        self.add_passthrough_option('--partition', type='string', default="group", help='the attribute in for each node which represents group membership')

    def mapper_init_sh(self):
        self.G = nx.read_gpickle(self.options.network)

        self.constraints = {}


    def mapper_init_b(self):
        self.G = nx.read_gpickle(self.options.network)

    def mapper_b(self, key, node):
        roles = dict((n, dict((role, 0) for role in brokerage._RoleClassifier.roleTypes)) for n in [node])

        for successor in self.G.successors(node):
            for predecessor in self.G.predecessors(node):
                if successor == predecessor or successor == node or predecessor == node: continue
                if not (self.G.has_edge(predecessor, successor)):
                    # found a broker!
                    # now which kind depends on who is in which group
                    roles[node][brokerage._RoleClassifier.classify(self.G.node[predecessor][self.options.partition], self.G.node[node][self.options.partition],
                                                                  self.G.node[successor][self.options.partition])] += 1
        yield None, roles

    def mapper_sh(self, _, node):

        sub = self.G.subgraph([x for x in self.G.neighbors(node)]+[node])
        self.A = nx.to_numpy_matrix(sub)
        self.p = structural_holes._calcProportionalTieStrengths(self.A)

        constraint = {"C-Index": 0.0, "C-Size": 0.0, "C-Density": 0.0, "C-Hierarchy": 0.0}

        Vi = structural_holes._neighborsIndexes(sub, node, self.options.includeOutLinks, self.options.includeInLinks)

        # i is the node we are calculating constraint for
        # and is thus the ego of the ego net
        i = sub.nodes().index(node)

        if not self.options.wholeNetwork:
            # need to recalculate p w/r/t this node
            pq = structural_holes._calcProportionaTieStrengthWRT(sub, i)
        else:
            # don't need to calculate p w/r/t any node
            pq = self.p

        for j in Vi:
            Pij = self.p[i, j]
            constraint["C-Size"] += Pij ** 2
            innerSum = 0.0
            for q in Vi:
                if q == j or q == i: continue
                innerSum += self.p[i, q] * pq[q, j]

            constraint["C-Hierarchy"] += innerSum ** 2
            constraint["C-Density"] += 2 * Pij * innerSum

        constraint["C-Index"] = constraint["C-Size"] + constraint["C-Density"] + constraint["C-Hierarchy"]
        yield None, {node:constraint}

    def reducer(self, key, values):
        z = {}
        for v in values:
            z.update(v)
        yield None, pd.DataFrame(z).to_json()

    def steps(self):
        if self.options.messure == "strutural-holes":
            return [
                MRStep(
                    mapper_init=self.mapper_init_sh,
                    mapper=self.mapper_sh,
                    reducer=self.reducer
                       )
            ]
        elif self.options.messure == "brokerage":
            return [
                MRStep(
                    mapper_init=self.mapper_init_b,
                    mapper=self.mapper_b,
                    reducer=self.reducer
                )
            ]


if __name__ == '__main__':
    MRJobStructuralHoles.run()
