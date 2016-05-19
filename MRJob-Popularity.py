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


class MRJobPopularity(MRJob):

    INPUT_PROTOCOL = JSONValueProtocol
    OUTPUT_PROTOCOL = JSONValueProtocol
    days = [7, 14, 30, 60, 90]

    def configure_options(self):
        super(MRJobPopularity, self).configure_options()

    def mapper(self, _, line):

        df = pd.read_json(line["raw"])
        df['time'] = df['time'].apply(dt)

        for d in self.days:
            dft = df.set_index(pd.DatetimeIndex(df['time']))
            start = dft.index.searchsorted(dft.index[0])
            end = dft.index.searchsorted(dft.index[0] + datetime.timedelta(days=d))
            dft = dft.ix[start:end]
            dftt = pd.DataFrame(index=dft.index)
            dftt["activations"] = 1
            dftt = dftt.resample('d', how='sum').fillna(0)
            dftt["activations"] = (dftt["activations"].cumsum() / dftt["activations"].sum())

            dftt["activations"].mean()
            yield (d, {"timedelta": d,
                                "popularity": dftt["activations"].mean(),
                                "word": line["file"].split("/")[-1]})

    def steps(self):
        return [MRStep(
               mapper=self.mapper
               )]


if __name__ == '__main__':
    MRJobPopularity.run()
