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
from mrjob.step import MRStep
import datetime


def dt(X):
    return datetime.datetime.fromtimestamp(float(X / 1000))


def to_date(X):
    return X.day()


class MRJobPopularityRaw(MRJob):

    INPUT_PROTOCOL = JSONValueProtocol
    OUTPUT_PROTOCOL = JSONValueProtocol
    days = [7, 14, 30, 60, 90]

    def configure_options(self):
        super(MRJobPopularityRaw, self).configure_options()
        self.add_passthrough_option('--avrage', type='int', default=0, help='...')

    def mapper(self, _, line):

        df = pd.read_json(line["raw"])
        df['time'] = df['time'].apply(dt)
        df = df.sort(["time"])
        df = df.resample('d').max().fillna(0)
        idx = pd.date_range(df.index[0], df.index[0] + datetime.timedelta(days=30))
        df = df.reindex(idx, fill_value=0)

        for k, v in dftt.reset_index().ix[self.days].iterrows():
            yield None, {"activation": v["activations"],
                                "word": line["file"].split("/")[-1],
                                "period": k}


    def reducer(self, key, values):
        d = []
        for v in values:
            d.append(v)

        yield None, pd.DataFrame(d).to_json()


    def steps(self):
        return [MRStep(
            mapper=self.mapper,
            reducer=self.reducer
               )]



if __name__ == '__main__':
    MRJobPopularityRaw.run()
