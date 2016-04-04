from __future__ import division

import random

import networkx as nx
import pandas as pd
from mrjob.job import MRJob
from mrjob.protocol import JSONValueProtocol
from mrjob.step import MRStep

import cascade
import metrics


class MRJobNetworkXSimulations(MRJob):

    OUTPUT_PROTOCOL = JSONValueProtocol
    INPUT_PROTOCOL = JSONValueProtocol

    def configure_options(self):
        super(MRJobNetworkXSimulations, self).configure_options()
        self.add_file_option('--network')

    def runCascade(self, C):
        cas = C
        idx = []
        values = []
        met = metrics.metric(cas.getGraph())
        while True:
            try:
                cas.next()
                met.add(cas.getInfectedNode())
                values.append(met.asMap())
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

        idx, values = self.runCascade(cascade.randomActive(self.G))
        idx1, values1 = self.runCascade(cascade.CascadeNabours(self.G))
        idx2, values2 = self.runCascade(cascade.NodeWithHighestActiveNabours(self.G))
        idx3, values3 = self.runCascade(cascade.NodeInSameCommunity(self.G))
        idx4, values4 = self.runCascade(cascade.CascadeNaboursWeight(self.G))

        df = pd.DataFrame(values, index=idx)
        df2 = pd.DataFrame(values1, index=idx1)
        df3 = pd.DataFrame(values2, index=idx2)
        df4 = pd.DataFrame(values3, index=idx3)
        df5 = pd.DataFrame(values4, index=idx4)

        result1_user = df.drop_duplicates(subset='numberActivatedUsers', keep='first').set_index(
            ['numberActivatedUsers'], verify_integrity=True).join(
            df2.drop_duplicates(subset='numberActivatedUsers', keep='first').set_index(['numberActivatedUsers'],
                                                                                       verify_integrity=True),
            how='outer', lsuffix="_m1", rsuffix="_m2")
        result2_user = df3.drop_duplicates(subset='numberActivatedUsers', keep='first').set_index(
            ['numberActivatedUsers'], verify_integrity=True).join(
            df4.drop_duplicates(subset='numberActivatedUsers', keep='first').set_index(['numberActivatedUsers'],
                                                                                       verify_integrity=True),
            how='outer', lsuffix="_m3", rsuffix="_m4")

        result3_user = df5.drop_duplicates(subset='numberActivatedUsers', keep='first').set_index(
            ['numberActivatedUsers'], verify_integrity=True)
        result_user = result1_user.join(result2_user, how='outer').join(result3_user, how='outer',
                                                                        rsuffix='_m5').fillna(method="ffill")

        result1_act = df.drop_duplicates(subset='numberOfActivations', keep='first').set_index(['numberOfActivations'],
                                                                                               verify_integrity=True).join(
            df2.drop_duplicates(subset='numberOfActivations', keep='first').set_index(['numberOfActivations'],
                                                                                      verify_integrity=True),
            how='outer', lsuffix="_m1", rsuffix="_m2")
        result2_act = df3.drop_duplicates(subset='numberOfActivations', keep='first').set_index(['numberOfActivations'],
                                                                                                verify_integrity=True).join(
            df4.drop_duplicates(subset='numberOfActivations', keep='first').set_index(['numberOfActivations'],
                                                                                      verify_integrity=True),
            how='outer', lsuffix="_m3", rsuffix="_m4")
        result3_act = df5.drop_duplicates(subset='numberOfActivations', keep='first').set_index(['numberOfActivations'],
                                                                                                verify_integrity=True)
        result_act = result1_act.join(result2_act, how='outer').join(result3_act, how='outer', rsuffix='_m5').fillna(
            method="ffill")

        if self.r_u_l is None and self.r_a_l is None:
            self.r_u_l = result_user
            self.r_a_l = result_act
        else:
            self.r_u_l = pd.concat((self.r_u_l, result_user))
            self.r_u_l = self.r_u_l.groupby(self.r_u_l.index).mean()

            self.r_a_l = pd.concat((self.r_a_l, result_act))
            self.r_a_l = self.r_a_l.groupby(self.r_a_l.index).mean()

    def mapper_final(self):
        if self.r_u_l is not None and self.r_a_l is not None:
            yield "tmp", {"result_user": self.r_u_l.to_json(orient='records'),
                          "result_act": self.r_a_l.to_json(orient='records')}

    def mapper_init_read(self):
        self.r_u_l = None
        self.r_a_l = None

    def mapper_read(self, _, v):
        if self.r_u_l is None:
            self.r_a_l = pd.read_json(v["result_act"])
            self.r_u_l = pd.read_json(v["result_user"])
        else:
            self.r_u_l = pd.concat((self.r_u_l, pd.read_json(v["result_user"])))
            self.r_u_l = self.r_u_l.groupby(self.r_u_l.index).mean()

            self.r_a_l = pd.concat((self.r_a_l, pd.read_json(v["result_act"])))
            self.r_a_l = self.r_a_l.groupby(self.r_a_l.index).mean()

    def combiner(self, key, values):
        r_u_l = None
        r_a_l = None
        for v in values:
            if r_u_l is None:
                r_a_l = pd.read_json(v["result_act"])
                r_u_l = pd.read_json(v["result_user"])
            else:
                r_u_l = pd.concat((r_u_l, pd.read_json(v["result_user"])))
                r_u_l = r_u_l.groupby(r_u_l.index).mean()

                r_a_l = pd.concat((r_a_l, pd.read_json(v["result_act"])))
                r_a_l = r_a_l.groupby(r_a_l.index).mean()

        yield key, {"result_user": r_u_l.to_json(orient='records'), "result_act": r_a_l.to_json(orient='records')}

    def reducer(self, key, values):
        r_u_l = None
        r_a_l = None
        for v in values:
            if r_u_l is None:
                r_a_l = pd.read_json(v["result_act"])
                r_u_l = pd.read_json(v["result_user"])
            else:
                r_u_l = pd.concat((r_u_l, pd.read_json(v["result_user"])))
                r_u_l = r_u_l.groupby(r_u_l.index).mean()

                r_a_l = pd.concat((r_a_l, pd.read_json(v["result_act"])))
                r_a_l = r_a_l.groupby(r_a_l.index).mean()

        yield key, {"result_user": r_u_l.to_json(orient='records'), "result_act": r_a_l.to_json(orient='records')}

    def steps(self):
        return [
            MRStep(mapper_init=self.mapper_init,
                   mapper=self.mapper,
                   mapper_final=self.mapper_final,
                   combiner=self.combiner,
                   reducer=self.reducer)
        ]


if __name__ == '__main__':
    MRJobNetworkXSimulations.run()
