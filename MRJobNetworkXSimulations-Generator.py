from __future__ import division

import networkx as nx
import pandas as pd
import random
from mrjob.job import MRJob
from mrjob.protocol import JSONValueProtocol
from mrjob.step import MRStep

import cascade


class MRJobNetworkXSimulations(MRJob):
    OUTPUT_PROTOCOL = JSONValueProtocol
    INPUT_PROTOCOL = JSONValueProtocol

    def configure_options(self):
        super(MRJobNetworkXSimulations, self).configure_options()
        self.add_file_option('--network')
        self.add_passthrough_option('--modle', type='int', default=0, help='...')
        self.add_passthrough_option('--itterations', type='int', default=20000, help='...')
        self.add_passthrough_option('--sampelFraction', type='int', default=10, help='...')
        self.add_passthrough_option('--resampeling', type='int', default=10, help='...')

    def runCascade(self, C):
        cas = C
        idx = []
        values = []
        while True:
            try:
                cas.next()
                values.append(cas.getInfectedNode())
                idx.append(cas.getStep())
            except StopIteration as err:
                break
            except Exception as e:
                print e
        return idx, values

    def mapper_init(self):

        self.G = nx.read_gpickle(self.options.network)
        nx.set_node_attributes(self.G, 'activated', {node: 0 for node in self.G.nodes()})
        seed = random.choice([n for n, attrdict in self.G.node.items() if attrdict['activated'] == 0])
        nx.set_node_attributes(self.G, 'activated', {seed: 1})

        self.r_u_l = None
        self.r_a_l = None

    def mapper(self, _, line):

        iteration = self.options.itterations
        sampelFraction = self.options.sampelFraction

        if self.options.modle == 0:
            idx, values = self.runCascade(cascade.randomActive(self.G, itterations=iteration))
        elif self.options.modle == 1:
            idx, values = self.runCascade(cascade.CascadeNabours(self.G, itterations=iteration))
        elif self.options.modle == 2:
            idx, values = self.runCascade(cascade.NodeWithHighestActiveNabours(self.G, itterations=iteration))
        elif self.options.modle == 3:
            idx, values = self.runCascade(cascade.NodeInSameCommunity(self.G, itterations=iteration))
        elif self.options.modle == 4:
            idx, values = self.runCascade(cascade.CascadeNaboursWeight(self.G, itterations=iteration))

        df = pd.DataFrame({"ids": values}, index=idx)

        for i in range(1, self.options.resampeling):
            print
            yield "tmp", df.sample(frac=(float(sampelFraction) / float(10))).to_json()

    def steps(self):
        return [
            MRStep(mapper_init=self.mapper_init,
                   mapper=self.mapper
                   )
        ]


if __name__ == '__main__':
    MRJobNetworkXSimulations.run()
