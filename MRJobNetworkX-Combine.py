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


# This cljob should only resutn the last row of each datafram and then cbine them, this is then the end point of each diffusion and not
# intermediate points as has been down with the avraging before.

class MRJobNetworkX(MRJob):
    OUTPUT_PROTOCOL = JSONValueProtocol

    def configure_options(self):
        super(MRJobNetworkX, self).configure_options()

    def mapper(self, _, line):
        yield "apple", line

    def combiner(self, key, values):
        r_u_l = None
        r_a_l = None
        for v in values:
            if r_u_l is None:
                r_a_l = pd.read_json(v["result_act"].tail(1))
                r_u_l = pd.read_json(v["result_user"].tail(1))
            else:
                r_u_l = pd.concat((r_u_l, pd.read_json(v["result_user"].tail(1))))
                r_u_l = r_u_l.groupby(r_u_l.index).mean()

                r_a_l = pd.concat((r_a_l, pd.read_json(v["result_act"].tail(1))))
                r_a_l = r_a_l.groupby(r_a_l.index).mean()

        yield key, {"result_user": r_u_l.to_json(orient='records'), "result_act": r_a_l.to_json(orient='records')}

    def reducer(self, key, values):
        r_u_l = None
        r_a_l = None
        for v in values:
            if r_u_l is None:
                r_a_l = pd.read_json(v["result_act"].tail(1))
                r_u_l = pd.read_json(v["result_user"].tail(1))
            else:
                r_u_l = pd.concat((r_u_l, pd.read_json(v["result_user"].tail(1))))
                r_u_l = r_u_l.groupby(r_u_l.index).mean()

                r_a_l = pd.concat((r_a_l, pd.read_json(v["result_act"].tail(1))))
                r_a_l = r_a_l.groupby(r_a_l.index).mean()

        yield key, {"result_user": r_u_l.to_json(orient='records'), "result_act": r_a_l.to_json(orient='records')}

    def steps(self):
        return [
            MRStep(mapper_init=self.mapper_init,
                   mapper=self.mapper,
                   combiner=self.combiner,
                   reducer=self.reducer
                   )
        ]


if __name__ == '__main__':
    MRJobNetworkX.run()
