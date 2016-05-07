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


class CascadeCount(MRJob):
    OUTPUT_PROTOCOL = JSONValueProtocol

    def configure_options(self):
        super(CascadeCount, self).configure_options()

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

                dtf = pd.read_csv(buf, index_col=False, header=None, sep="\t", engine="python",
                                  compression=None).drop_duplicates(subset=[2], keep='last')
                yield "apple", len(dft.index)

    def steps(self):
        return [
            MRStep(
                   mapper=self.mapper
                   )
        ]


if __name__ == '__main__':
    CascadeCount.run()
