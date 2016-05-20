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
        self.add_passthrough_option('--avrage', type='int', default=0, help='...')

    def mapper(self, _, line):

        df = pd.read_json(line["raw"])
        df['time'] = df['time'].apply(dt)
        df = df.sort(["time"])

        for d in self.days:
            dft = df.set_index(pd.DatetimeIndex(df['time']))

            start = dft.index.searchsorted(dft.index[0])
            end = dft.index.searchsorted(dft.index[0] + datetime.timedelta(days=d))
            dft = dft.ix[start:end]
            dftt = pd.DataFrame(index=dft.index)

            dftt["activations"] = 1
            dftt = dftt.resample('d').sum().fillna(0)
            idx = pd.date_range(dftt.index[0], dftt.index[0] + datetime.timedelta(days=d))
            dftt = dftt.reindex(idx, fill_value=0)
            dftt["activations"] = (dftt["activations"].cumsum() / dftt["activations"].sum())
            dftt.index.name = 'date'
            if self.options.avrage == 1:
                yield None, {"timedelta": d,
                                    "popularity": dftt["activations"].mean(),
                                    "activations": dftt["activations"].to_json(),
                                    "word": line["file"].split("/")[-1]}
            else:
                yield d, {"timedelta": d,
                             "popularity": dftt["activations"].mean(),
                             "activations": dftt.reset_index().to_json(),
                             "word": line["file"].split("/")[-1]}

    def reducer(self, key, values):
        d = []
        for v in values:
            d.append(v)

        yield None, pd.DataFrame(d).to_json()

    def reducer_2(self, key, values):
        df = None
        for v in values:
            if df is None:
                df = pd.read_json(v["activations"])
            else:
                pd.concat((df, pd.read_json(v["activations"])))

        yield key, {"D":key, "vales":df.groupby(df["date"]).mean().to_json()}

    def steps(self):
        if self.options.avrage == 1:
            return [MRStep(
                mapper=self.mapper,
                reducer=self.reducer
                   )]
        else:
            return [MRStep(
                mapper=self.mapper,
                reducer=self.reducer_2
                   )]


if __name__ == '__main__':
    MRJobPopularity.run()
