from __future__ import division

import gzip

try:
    from BytesIO import BytesIO
except ImportError:
    from io import BytesIO

try:
    from urlparse import urlparse
except ImportError:
    from urllib.parse import urlparse

import hdfs
import networkx as nx
import pandas as pd
from mrjob.job import MRJob
from mrjob.protocol import JSONValueProtocol
from mrjob.step import MRStep

import cascade
import metrics
import json

class MRJobNetworkX(MRJob):
    OUTPUT_PROTOCOL = JSONValueProtocol

    def configure_options(self):
        super(MRJobNetworkX, self).configure_options()
        self.add_file_option('--network')
        self.add_passthrough_option('--avrage', type='int', default=0, help='...')

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
        nx.set_node_attributes(self.G, 'activated', self.tmp)
        client = hdfs.client.Client("http://" + urlparse(line).netloc)
        if line[-1] != "#":
            with client.read(urlparse(line).path) as r:
                # with open(urlparse(line).path) as r:
                buf = BytesIO(r.read())
                if ".gz" in line:
                    gzip_f = gzip.GzipFile(fileobj=buf)
                    content = gzip_f.read()
                    idx, values = self.runCascade(cascade.actualCascade(StringIO.StringIO(content), self.G))
                else:
                    idx, values = self.runCascade(cascade.actualCascade(buf, self.G))
                df = pd.DataFrame(values, index=idx)
                result_user = df.drop_duplicates(subset='numberActivatedUsers', keep='first').set_index(
                    ['numberActivatedUsers'], verify_integrity=True)
                result_act = df.drop_duplicates(subset='numberOfActivations', keep='first').set_index(
                    ['numberOfActivations'], verify_integrity=True)

            yield "apple", {"file": line, "name": line.split("/")[-1],
                            "result_user": result_user.loc[-1:].to_json(orient='records'),
                            "result_act": result_act.loc[-1:].to_json(orient='records')}

    def mappertwo(self, _, line):
        yield "apple", json.loads(line)

    def reducer(self, key, values):
        r_u_l = None
        r_a_l = None
        for v in values:
            print v

            if r_u_l is None:
                r_a_l = pd.read_json(v["result_act"])
                r_u_l = pd.read_json(v["result_user"])
            else:
                r_u_l = pd.concat((r_u_l, pd.read_json(v["result_user"])))
                r_a_l = pd.concat((r_a_l, pd.read_json(v["result_act"])))

        r_u_l = r_u_l.groupby(r_u_l.index).mean()
        r_a_l = r_a_l.groupby(r_a_l.index).mean()

        yield key, {"result_user": r_u_l.to_json(), "result_act": r_a_l.to_json()}

    def steps(self):
        if self.options.avrage == 1:
            return [
                MRStep(
                    mapper=self.mappertwo,
                       reducer=self.reducer
                       )
            ]
        else:
            return [
                MRStep(
                    mapper=self.mappertwo
                       )
            ]
if __name__ == '__main__':
    MRJobNetworkX.run()
