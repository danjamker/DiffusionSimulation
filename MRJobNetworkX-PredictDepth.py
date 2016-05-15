from __future__ import division

try:
    from BytesIO import BytesIO
except ImportError:
    from io import BytesIO

try:
    from urlparse import urlparse
except ImportError:
    from urllib.parse import urlparse

import networkx as nx
import pandas as pd
from mrjob.job import MRJob
from mrjob.protocol import JSONValueProtocol
from mrjob.step import MRStep

import metrics


class MRJobNetworkX(MRJob):
    OUTPUT_PROTOCOL = JSONValueProtocol
    INPUT_PROTOCOL = JSONValueProtocol

    def configure_options(self):
        super(MRJobNetworkX, self).configure_options()
        self.add_file_option('--network')
        self.add_passthrough_option('--avrage', type='int', default=0, help='...')
        self.add_passthrough_option('--limit', type='int', default=50, help='...')

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
            except StopIteration:
                break
        return idx, values

    def mapper_init(self):
        self.G = nx.read_gpickle(self.options.network)
        self.tmp = {node: 0 for node in self.G.nodes()}
        nx.set_node_attributes(self.G, 'activated', self.tmp)

    def mapper(self, _, line):
        df = pd.read_json(json.loads(line)["raw"])
        result_act = pd.read_json(json.loads(line)["result_act"])
        result_user = pd.read_json(json.loads(line)["result_user"])

        if len(df.index) > self.options.limit:
            for r in range(50, 100):
                result_user_100 = df.loc[:r].drop_duplicates(subset='numberActivatedUsers',
                                                                              keep='first').set_index(
                    ['numberActivatedUsers'], verify_integrity=True, drop=False)
                result_act_100 = df.loc[:r].drop_duplicates(subset='numberOfActivations',
                                                                             keep='first').set_index(
                    ['numberOfActivations'], verify_integrity=True, drop=False)


                ruy = result_user_100.iloc[-1:]
                ruy.index = [len(result_user.index)]
                ray = result_act_100.iloc[-1:]
                ray.index = [len(result_act.index)]
                ray["depth"] = [len(result_act.index)]
                ruy["depth"] = [len(result_user.index)]
                ray["word"] = [line["file"].split("/")[-1]]
                ruy["word"] = [line["file"].split("/")[-1]]

                yield r, {"name": line["file"].split("/")[-1],
                                "result_user": ruy.to_json(),
                                "result_act": ray.to_json()}

    def reducer(self, key, values):
        r_u_l = None
        r_a_l = None
        for v in values:
            if r_u_l is None:
                r_a_l = pd.read_json(v["result_act"])
                r_u_l = pd.read_json(v["result_user"])
            else:
                r_u_l = pd.concat((r_u_l, pd.read_json(v["result_user"])))
                r_a_l = pd.concat((r_a_l, pd.read_json(v["result_act"])))

        yield key, {"observation_level": key, "result_user": r_u_l.reset_index().to_json(),
                    "result_act": r_a_l.reset_index().to_json()}

    def steps(self):
        return [
            MRStep(mapper_init=self.mapper_init,
                   mapper=self.mapper,
                   reducer=self.reducer
                   )
        ]


if __name__ == '__main__':
    MRJobNetworkX.run()