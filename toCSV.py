from __future__ import division

try:
    from BytesIO import BytesIO
except ImportError:
    from io import BytesIO

try:
    from urlparse import urlparse
except ImportError:
    from urllib.parse import urlparse

import pandas as pd

from mrjob.job import MRJob
from mrjob.protocol import JSONValueProtocol
from mrjob.protocol import TextProtocol
from mrjob.step import MRStep

class toCSV(MRJob):

    INPUT_PROTOCOL = JSONValueProtocol
    OUTPUT_PROTOCOL = TextProtocol

    def mapper(self, key, value):
        df = pd.read_json(value["raw"])

        for index, row in df.iterrows():
            v = [index, value["name"]] + list(row.values)
            yield None, ','.join([str(i) for i in v])

    def steps(self):
        return [MRStep(
            mapper=self.mapper
               )]


if __name__ == '__main__':
    toCSV.run()
