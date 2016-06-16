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
from sklearn.metrics import roc_curve, auc


def dt(X):
    return datetime.datetime.fromtimestamp(float(X / 1000))


def to_date(X):
    return X.day()


class MRROC(MRJob):

    OUTPUT_PROTOCOL = JSONValueProtocol

    def configure_options(self):
        super(MRROC, self).configure_options()

    def mapper(self, key, value):

        values = value.split(",")

        yield {"username":values[1],
               "model":values[0]}, {"word":values[3],
                                    "performed":values[4],
                                    "value":values[6]}

    def reducer(self, key, values):

        df = []
        for v in values:
            df.append(v)
        df = pd.DataFrame(df)

        fpr, tpr, thr = roc_curve(df["performed"].values, df["value"].values, pos_label=1)

        yield None, {"username":key["username"],
                     "model":key["model"],
                     "fpr":fpr,
                     "tpr":tpr,
                     "threshold":thr}



    def steps(self):
        return [MRStep(
            mapper=self.mapper,
            reducer=self.reducer_kmean
               )]

if __name__ == '__main__':
    MRROC.run()