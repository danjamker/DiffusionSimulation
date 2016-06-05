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
import networkx as nx

class MRJobStructuralHoles(MRJob):
    OUTPUT_PROTOCOL = JSONValueProtocol

    def configure_options(self):
        super(MRJobStructuralHoles, self).configure_options()

        self.add_file_option('--network')

        self.add_passthrough_option('--wholeNetwork', type='int', default=1, help='...')
        self.add_passthrough_option('--includeOutLinks', type='int', default=1, help='...')
        self.add_passthrough_option('--includeInLinks', type='int', default=0, help='...')

    def mapper_init(self):
        self.G = nx.read_gpickle(self.options.network)
        self.A = nx.to_numpy_matrix(self.G)
        self.p = structural_holes._calcProportionalTieStrengths(A)
        self.constraints = {}

    def mapper(self, _, node):
        constraint = {"C-Index": 0.0, "C-Size": 0.0, "C-Density": 0.0, "C-Hierarchy": 0.0}

        Vi = structural_holes._neighborsIndexes(self.G, node, self.options.includeOutLinks, self.options.includeInLinks)

        # i is the node we are calculating constraint for
        # and is thus the ego of the ego net
        i = self.G.nodes().index(node)

        if not self.options.wholeNetwork:
            # need to recalculate p w/r/t this node
            pq = structural_holes._calcProportionaTieStrengthWRT(A, i)
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
        yield node, constraint

    def steps(self):
        return [
            MRStep(
                mapper=self.mapper
                   )
        ]


if __name__ == '__main__':
    MRJobStructuralHoles.run()
