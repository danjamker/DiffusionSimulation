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
import networkx as nx
import pandas as pd
from mrjob.job import MRJob
from mrjob.protocol import JSONValueProtocol
from mrjob.step import MRStep

import cascade
import metrics
import numpy as np

class MRJobNetworkX(MRJob):
    OUTPUT_PROTOCOL = JSONValueProtocol

    def configure_options(self):
        super(MRJobNetworkX, self).configure_options()
        self.add_file_option('--network')
        self.add_passthrough_option('--avrage', type='int', default=0, help='...')

    def runCascade(self, C):
        cas = C
        idx = []
        values = []
        met = metrics.metric(cas.getGraph())
        while True:
            try:
                cas.next()
                met.add(cas.getInfectedNode(), cas.getStepTime())
                values.append(met.asMap())
                idx.append(cas.getStep())
            except StopIteration:
                break
        return idx, values

    def mapper_init(self):
        self.G = nx.read_gpickle(self.options.network)
        self.tmp = {node: 0 for node in self.G.nodes()}
        nx.set_node_attributes(self.G, 'activated', self.tmp)

    def mapper(self, _, line):

        nx.set_node_attributes(self.G, 'activated', self.tmp)
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

                idx, values = self.runCascade(cascade.actualCascade(buf, self.G))
                df = pd.DataFrame(values, index=idx).sort_index()

                #check to see if there is data within the dataframes.
                if len(df.index) > 0:
                    # Set the index to tbe the number of active users, this is then at the first instance that
                    #A new user uses the actions
                    result_user = df.drop_duplicates(subset='numberActivatedUsers', keep='first').set_index(
                        ['numberActivatedUsers'], verify_integrity=True).sort_index()

                    result_user["surface_mean"] = result_user["surface"].expanding(min_periods=1).mean()
                    result_user["surface_cv"] = result_user["surface"].expanding(min_periods=1).std()
                    result_user["surface_var"] = result_user["surface"].expanding(min_periods=1).var()

                    result_user["degree_mean"] = result_user["degree"].expanding(min_periods=1).mean()
                    result_user["degree_median"] = result_user["degree"].expanding(min_periods=1).median()
                    result_user["degree_cv"] = result_user["degree"].expanding(min_periods=1).std()
                    result_user["degree_var"] = result_user["degree"].expanding(min_periods=1).var()
                    result_user["degree_max"] = result_user["degree"].expanding(min_periods=1).max()
                    result_user["degree_min"] = result_user["degree"].expanding(min_periods=1).min()

                    result_user["UserExposure_mean"] = result_user["UserExposure"].expanding(min_periods=1).mean()
                    result_user["UserExposure_cv"] = result_user["UserExposure"].expanding(min_periods=1).std()
                    result_user["UserExposure_var"] = result_user["UserExposure"].expanding(min_periods=1).var()
                    result_user["UserExposure_median"] = result_user["UserExposure"].expanding(min_periods=1).median()
                    result_user["UserExposure_min"] = result_user["UserExposure"].expanding(min_periods=1).max()
                    result_user["UserExposure_mean"] = result_user["UserExposure"].expanding(min_periods=1).min()

                    result_user["pagerank_mean"] = result_user["pagerank"].expanding(min_periods=1).mean()
                    result_user["pagerank_cv"] = result_user["pagerank"].expanding(min_periods=1).std()
                    result_user["pagerank_var"] = result_user["pagerank"].expanding(min_periods=1).var()
                    result_user["pagerank_median"] = result_user["pagerank"].expanding(min_periods=1).median()
                    result_user["pagerank_min"] = result_user["pagerank"].expanding(min_periods=1).max()
                    result_user["pagerank_mean"] = result_user["pagerank"].expanding(min_periods=1).min()

                    result_user["time_step"] = result_user["time"].diff()
                    result_user["time_step_mean"] = (result_user["time_step"] / np.timedelta64(1, 's')).expanding(
                        min_periods=1).mean()
                    result_user["time_step_cv"] = (result_user["time_step"] / np.timedelta64(1, 's')).expanding(
                        min_periods=1).std()
                    result_user["time_step_median"] = (result_user["time_step"] / np.timedelta64(1, 's')).expanding(
                        min_periods=1).median()
                    result_user["time_step_min"] = (result_user["time_step"] / np.timedelta64(1, 's')).expanding(
                        min_periods=1).min()
                    result_user["time_step_max"] = (result_user["time_step"] / np.timedelta64(1, 's')).expanding(
                        min_periods=1).max()
                    result_user["time_step_var"] = (result_user["time_step"] / np.timedelta64(1, 's')).expanding(
                        min_periods=1).var()


                    result_act = df.drop_duplicates(subset='numberOfActivations', keep='first').set_index(
                        ['numberOfActivations'], verify_integrity=True).sort_index()
                    result_act["surface_mean"] = result_act["surface"].expanding(min_periods=1).mean()
                    result_act["surface_cv"] = result_act["surface"].expanding(min_periods=1).std()
                    result_act["surface_var"] = result_act["surface"].expanding(min_periods=1).var()

                    result_act["degree_mean"] = result_act["degree"].expanding(min_periods=1).mean()
                    result_act["degree_median"] = result_act["degree"].expanding(min_periods=1).median()
                    result_act["degree_cv"] = result_act["degree"].expanding(min_periods=1).std()
                    result_act["degree_var"] = result_act["degree"].expanding(min_periods=1).var()
                    result_act["degree_max"] = result_act["degree"].expanding(min_periods=1).max()
                    result_act["degree_min"] = result_act["degree"].expanding(min_periods=1).min()

                    result_act["ActivateionExposure_mean"] = result_act["ActivateionExposure"].expanding(
                        min_periods=1).mean()
                    result_act["ActivateionExposure_cv"] = result_act["ActivateionExposure"].expanding(
                        min_periods=1).std()
                    result_act["ActivateionExposure_var"] = result_act["ActivateionExposure"].expanding(
                        min_periods=1).var()
                    result_act["ActivateionExposure_var"] = result_act["ActivateionExposure"].expanding(
                        min_periods=1).var()
                    result_act["ActivateionExposure_median"] = result_act["ActivateionExposure"].expanding(
                        min_periods=1).median()
                    result_act["ActivateionExposure_max"] = result_act["ActivateionExposure"].expanding(
                        min_periods=1).max()
                    result_act["ActivateionExposure_min"] = result_act["ActivateionExposure"].expanding(
                        min_periods=1).min()

                    result_act["pagerank_mean"] = result_act["pagerank"].expanding(min_periods=1).mean()
                    result_act["pagerank_cv"] = result_act["pagerank"].expanding(min_periods=1).std()
                    result_act["pagerank_var"] = result_act["pagerank"].expanding(min_periods=1).var()
                    result_act["pagerank_var"] = result_act["pagerank"].expanding(min_periods=1).var()
                    result_act["pagerank_median"] = result_act["pagerank"].expanding(min_periods=1).median()
                    result_act["pagerank_max"] = result_act["pagerank"].expanding(min_periods=1).max()
                    result_act["pagerank_min"] = result_act["pagerank"].expanding(min_periods=1).min()

                    result_act["time_step"] = result_act["time"].diff()
                    result_act["time_step_mean"] = (result_act["time_step"] / np.timedelta64(1, 's')).expanding(
                        min_periods=1).mean()
                    result_act["time_step_cv"] = (result_act["time_step"] / np.timedelta64(1, 's')).expanding(
                        min_periods=1).std()
                    result_act["time_step_median"] = (result_act["time_step"] / np.timedelta64(1, 's')).expanding(
                        min_periods=1).median()
                    result_act["time_step_min"] = (result_act["time_step"] / np.timedelta64(1, 's')).expanding(
                        min_periods=1).min()
                    result_act["time_step_max"] = (result_act["time_step"] / np.timedelta64(1, 's')).expanding(
                        min_periods=1).max()
                    result_act["time_step_var"] = (result_act["time_step"] / np.timedelta64(1, 's')).expanding(
                        min_periods=1).var()

                    yield "apple", {"file": line, "name": line.split("/")[-1],
                                    "result_user": result_user.to_json(),
                                    "result_act": result_act.to_json(),
                                    "raw": df.sort_index().reset_index().to_json()}

    def combiner(self, key, values):
        r_u_l = None
        r_a_l = None
        for v in values:
            if r_u_l is None:
                r_a_l = pd.read_json(v["result_act"])
                r_u_l = pd.read_json(v["result_user"])
            else:
                r_u_l = pd.concat((r_u_l, pd.read_json(v["result_user"])))
                r_u_l = r_u_l.groupby(r_u_l.index).mean()

                r_a_l = pd.concat((r_a_l, pd.read_json(v["result_act"])))
                r_a_l = r_a_l.groupby(r_a_l.index).mean()

        yield key, {"result_user": r_u_l.to_json(), "result_act": r_a_l.to_json()}

    def reducer(self, key, values):
        r_u_l = None
        r_a_l = None
        for v in values:
            if r_u_l is None:
                r_a_l = pd.read_json(v["result_act"])
                r_u_l = pd.read_json(v["result_user"])
            else:
                r_u_l = pd.concat((r_u_l, pd.read_json(v["result_user"])))
                r_u_l = r_u_l.groupby(r_u_l.index).mean()

                r_a_l = pd.concat((r_a_l, pd.read_json(v["result_act"])))
                r_a_l = r_a_l.groupby(r_a_l.index).mean()

        yield key, {"result_user": r_u_l.to_json(), "result_act": r_a_l.to_json()}

    def steps(self):
        if self.options.avrage == 1:
            return [
                MRStep(mapper_init=self.mapper_init,
                       mapper=self.mapper,
                       combiner=self.combiner,
                       reducer=self.reducer
                       )
            ]
        else:
            return [
                MRStep(mapper_init=self.mapper_init,
                       mapper=self.mapper
                       )
            ]
if __name__ == '__main__':
    MRJobNetworkX.run()
