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

class MRJobNetworkX(MRJob):

    OUTPUT_PROTOCOL = JSONValueProtocol

    def configure_options(self):
        super(MRJobNetworkX, self).configure_options()
        self.add_file_option('--network')

    def runCascade(self, C):
        cas = C
        idx = []
        values = []
        met = metrics.metric(cas.getGraph(), time_format="%Y-%m-%d")
        while True:
            try:
                cas.next()
                met.add(cas.getInfectedNode(), cas.getStepTime(), cas.getTag())
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
                buf = BytesIO(r.read())

                # If the data is in a GZipped file.
                if ".gz" in line:
                    gzip_f = gzip.GzipFile(fileobj=buf)
                    content = gzip_f.read()
                    buf = StringIO.StringIO(content)

                idx, values = self.runCascade(cascade.actualCascade(buf, self.G))
                df = pd.DataFrame(values, index=idx).sort_index()

                if len(df.index) > 0:
                    yield None, {"file": line, "name": line.split("/")[-1],
                                    "raw": df.sort_index().reset_index().to_json()}

    def steps(self):
        return [
            MRStep(mapper_init=self.mapper_init,
                   mapper=self.mapper
                   )
        ]

if __name__ == '__main__':
    MRJobNetworkX.run()
