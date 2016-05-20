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

import metrics
import numpy as np

import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

class MRJobNetworkX(MRJob):

    OUTPUT_PROTOCOL = JSONValueProtocol
    INPUT_PROTOCOL = JSONValueProtocol

    combinations = {
        "time":["time_step_mean","time_step_cv"],
        "basic":["surface","numberActivatedUsersnorm"],
        "community":["inffectedCommunitiesnor","usageEntorpy","userUsageEntorpy"],
        "exposure":["UserExposure_mean", "ActivateionExposure_mean"],
        "all":["time_step_mean","time_step_cv","surface","numberActivatedUsersnorm","inffectedCommunitiesnor","usageEntorpy","userUsageEntorpy"]
    }

    def configure_options(self):
        super(MRJobNetworkX, self).configure_options()
        self.add_passthrough_option('--lower_limit', type='int', default=50, help='...')

    def runCascade(self, C):
        cas = C
        idx = []
        values = []
        met = metrics.metric(cas.getGraph())
        while True:
            try:
                cas.next()
                met.add(cas.getInfectedNode())
                values.append(met.asMap())
                idx.append(cas.getStep())
            except StopIteration:
                break
        return idx, values

    def mapper(self, _, line):
        df = pd.read_json(line["raw"])
        df['time'] = pd.to_datetime(df['time'])

        result_act = pd.read_json(line["result_act"])
        result_user = pd.read_json(line["result_user"])

        if len(df.index) > self.options.lower_limit:
            #Check to see if are enough records in the range.
            for z in range(2, len(df)-2):
                for r in range(2, len(df)):
                    result_act_100, result_user_100 = self.generate_tables(df.loc[:r])
                    ruy = result_user_100.iloc[-1:]
                    ruy.index = [len(result_user.index)]
                    ray = result_act_100.iloc[-1:]
                    ray.index = [len(result_act.index)]
                    ray["depth"] = [len(result_act.index)]
                    ruy["depth"] = [len(result_user.index)]
                    ray["word"] = [line["file"].split("/")[-1]]
                    ruy["word"] = [line["file"].split("/")[-1]]

                    yield r, {"name": line["file"].split("/")[-1],
                                    "result_user": ruy.to_json(),
                                    "result_act": ray.to_json()}

    def generate_tables(self, df):
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
        result_user["ActivateionExposure_mean"] = result_user["ActivateionExposure"].expanding(
            min_periods=1).mean()
        result_user["ActivateionExposure_cv"] = result_user["ActivateionExposure"].expanding(
            min_periods=1).std()
        result_user["ActivateionExposure_var"] = result_user["ActivateionExposure"].expanding(
            min_periods=1).var()
        result_user["ActivateionExposure_var"] = result_user["ActivateionExposure"].expanding(
            min_periods=1).var()
        result_user["ActivateionExposure_median"] = result_user["ActivateionExposure"].expanding(
            min_periods=1).median()
        result_user["ActivateionExposure_max"] = result_user["ActivateionExposure"].expanding(
            min_periods=1).max()
        result_user["ActivateionExposure_min"] = result_user["ActivateionExposure"].expanding(
            min_periods=1).min()
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
        result_act["UserExposure_mean"] = result_act["UserExposure"].expanding(min_periods=1).mean()
        result_act["UserExposure_cv"] = result_act["UserExposure"].expanding(min_periods=1).std()
        result_act["UserExposure_var"] = result_act["UserExposure"].expanding(min_periods=1).var()
        result_act["UserExposure_median"] = result_act["UserExposure"].expanding(min_periods=1).median()
        result_act["UserExposure_min"] = result_act["UserExposure"].expanding(min_periods=1).max()
        result_act["UserExposure_mean"] = result_act["UserExposure"].expanding(min_periods=1).min()
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
        return result_act, result_user

    def reducer(self, key, values):

        r_u_l = None
        r_a_l = None
        for v in values:
            if r_u_l is None:
                r_a_l = pd.read_json(v["result_act"])
                r_u_l = pd.read_json(v["result_user"])
            else:
                r_u_l = pd.concat((r_u_l, pd.read_json(v["result_user"])))
                r_a_l = pd.concat((r_a_l, pd.read_json(v["result_act"])))

        if len(r_a_l) > 1:
            for k, v in self.combinations.iteritems():
                # r_u_l_results = self.liniar_regression(r_u_l, features=v)
                r_a_l_results = self.liniar_regression(r_a_l, features=v)

                yield None, {"observation_level": key, "r_u_l_results": "",
                        "r_a_l_results": r_a_l_results, "combination":k}

    def liniar_regression(self, df, features = [], test_size = 0.20, random_state = 0):
        X_train, X_test, Y_train, Y_test = sklearn.cross_validation.train_test_split(df[features],
                                                                                     df["depth"],
                                                                                     test_size=test_size, random_state=random_state)
        lm = LinearRegression()
        lm.fit(X_train, Y_train)

        mse_y_train = mean_squared_error(Y_train, lm.predict(X_train))
        mse_x_y_test = mean_squared_error(Y_test, lm.predict(X_test))

        return mse_x_y_test, mse_y_train

    def steps(self):
        return [
            MRStep(
                   mapper=self.mapper,
                   reducer=self.reducer
                   )
        ]


if __name__ == '__main__':
    MRJobNetworkX.run()
