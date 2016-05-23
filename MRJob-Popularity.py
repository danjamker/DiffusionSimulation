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
    INTERNAL_PROTOCOL = JSONValueProtocol
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
            dft = dft.resample('d').max()
            idx = pd.date_range(dft.index[0], dft.index[0] + datetime.timedelta(days=d))
            dft = dft.reindex(idx, fill_value=0, method='ffill').fillna(method='ffill')

            dft["user_popularity"] = (dft["number_activated_users"]/ dft["number_activated_users"].sum())
            dft["popularity"] = (dft["number_activations"]/ dft["number_activations"].sum())

            if self.options.avrage == 0:
                yield None, {"timedelta": d,
                                    "popularity": dft["popularity"].mean(),
                                    "popularity_json": dft[["popularity"]].reset_index().to_json(),
                                    "user_popularity": dft["user_popularity"].mean(),
                                    "user_popularity_json": dft[["user_popularity"]].reset_index().to_json(),
                                    "word": line["file"].split("/")[-1]}
            else:
                yield d, {"timedelta": d,
                          "popularity": dft["popularity"].mean(),
                          "popularity_json": dft.reset_index()[["popularity"]].to_json(),
                          "user_popularity": dft["user_popularity"].mean(),
                          "user_popularity_json": dft.reset_index()[["user_popularity"]].to_json(),
                          "word": line["file"].split("/")[-1]}

    def reducer(self, key, values):
        d = []
        for v in values:
            d.append(v)

        yield None, pd.DataFrame(d).to_json()

    def reducer_2(self, key, values):
        df_popularity = None
        df_user_popularity = None
        for v in values:
            if df_popularity is None:
                print v["popularity_json"]
                df_popularity = pd.read_json(v["popularity_json"])
                df_user_popularity = pd.read_json(v["user_popularity_json"])
            else:
                df_popularity = pd.concat((df_popularity, pd.read_json(v["popularity_json"])))
                df_user_popularity = pd.concat((df_user_popularity, pd.read_json(v["user_popularity_json"])))

        yield key, {"period":key, "popularity":df_popularity.groupby(df_popularity.index).mean().to_json(), "user_popularity":df_user_popularity.groupby(df_user_popularity.index).mean().to_json()}

    def steps(self):
        if self.options.avrage == 0:
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
