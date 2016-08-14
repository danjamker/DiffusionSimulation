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
import json

from sklearn import linear_model

class ConstraintCorrolation(MRJob):

    INPUT_PROTOCOL = JSONValueProtocol
    OUTPUT_PROTOCOL = JSONValueProtocol

    def mapper(self, key, value):
        df = pd.read_json(value["raw"])

        for row in df.iterrows():

            final = 0
            #final is the final size of the cascade
            for i in range(0, final):
                x = 0
                y = 0

                yield i, (x, y)

    def mapper_final(self, key, value):
        df = pd.read_json(value["raw"])
        #i is the final size,
        yield i, (x, y)

    def reducer(self, key, values):

        v1_array = []
        v2_array = []

        for v1, v2 in values:
            v1_array.append(v1)
            v2_array.append(v2)

        regr = linear_model.LinearRegression()

        # Train the model using the training sets
        regr.fit(v1_array, v2_array)

        yield key, regr.coef_

    def steps(self):
        return [MRStep(
            mapper=self.mapper,
            reducer=self.reducer
               )]


if __name__ == '__main__':
    ConstraintCorrolation.run()
