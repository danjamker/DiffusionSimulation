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
import pandas as pd
from mrjob.job import MRJob
from mrjob.protocol import JSONValueProtocol
from mrjob.step import MRStep

import metrics
import cascade

from networkx_additional_algorithms import brokerage

class MRJobCascade(MRJob):
    OUTPUT_PROTOCOL = JSONValueProtocol

    def configure_options(self):
        super(MRJobCascade, self).configure_options()

    def runCascade(self, C):
        cas = C
        idx = []
        values = []
        met = metrics.broker_metrics(cas.getGraph())
        while True:
            try:
                cas.next()
                met.add(cas.getInfectedNode())
                values.append(met.asMap())
                idx.append(cas.getStep())
            except StopIteration:
                break
        return idx, values

    def mapper(self, _, line):
        client = hdfs.client.Client("http://" + urlparse(line).netloc)

        if line[-1] != "#":
            with client.read(urlparse(line).path) as r:
                # with open(urlparse(line).path) as r:
                buf = BytesIO(r.read())

                # If the data is in a GZipped file.
                if ".gz" in line:
                    gzip_f = gzip.GzipFile(fileobj=buf)
                    content = gzip_f.read()
                    buf = StringIO.StringIO(content)

        cas = cascade.actualCascade(buf, self.G)
        idx, values = self.runCascade(cas)

    def steps(self):
        return [
            MRStep(
                   mapper=self.mapper
                   )
        ]


if __name__ == '__main__':
    MRJobCascade.run()
