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

import sklearn
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def dt(X):
    return datetime.datetime.fromtimestamp(float(X / 1000))


def to_date(X):
    return X.day()


class MRJobPopularityRaw(MRJob):

    INPUT_PROTOCOL = JSONValueProtocol
    OUTPUT_PROTOCOL = JSONValueProtocol

    combinations = {
        "time":["time_step_mean","time_step_cv"],
        "basic":["surface","numberActivatedUsersnorm"],
        "community":["inffectedCommunitiesnor","usageEntorpy","userUsageEntorpy"],
        "exposure":["UserExposure_mean", "ActivateionExposure_mean"],
        "all":["time_step_mean","time_step_cv","surface","numberActivatedUsersnorm","inffectedCommunitiesnor","usageEntorpy","userUsageEntorpy"]
    }

    target = ["user_target","activation_target"]

    def configure_options(self):
        super(MRJobPopularityRaw, self).configure_options()
        self.add_passthrough_option('--avrage', type='int', default=0, help='...')

    def mapper(self, _, line):

        df = pd.read_json(line["raw"])
        dfu, df = self.generate_tables(df)

        df['time'] = df['time'].apply(dt)
        df = df.set_index(pd.DatetimeIndex(df['time']))

        df = df.resample('d').max()
        idx = pd.date_range(df.index[0], df.index[0] + datetime.timedelta(days=30))
        df = df.reindex(idx, fill_value=0, method='ffill').fillna(method='ffill')

        df["user_target"] = df["numberActivatedUsers"].values[-1]
        df["activation_target"] = df["numberOfActivations"].values[-1]

        for k, v in df.reset_index().iterrows():
            yield k, {"df": v.to_json(),
                                "word": line["file"].split("/")[-1],
                                "period": 30}


    def reducer(self, key, values):

        df = pd.DataFrame()
        for v in values:
            df = df.append(pd.read_json(v["df"], typ='series', dtype=False), ignore_index=True)

        if len(df) > 1:
            for k, v in self.combinations.iteritems():
                for t in self.target:
                    #TODO loop over this 10 times to calculate the mean and variance.
                    results_array = []
                    for x in range(0,10):
                        results_array.append(self.liniar_regression(df.fillna(0), features=v, target=t)[1])

                    yield None, {"observation_level": key, "result_mean": np.mean(results_array),  "results_var": np.var(results_array), "combination":k, "target":t}

    def generate_tables(self, df):
        result_user = df.drop_duplicates(subset='numberActivatedUsers', keep='first').set_index(
            ['numberActivatedUsers'], verify_integrity=True, drop=False).sort_index()
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
        result_user["time_step_mean"] = (result_user["time_step"]).expanding(
            min_periods=1).mean()
        result_user["time_step_cv"] = (result_user["time_step"]).expanding(
            min_periods=1).std()
        result_user["time_step_median"] = (result_user["time_step"]).expanding(
            min_periods=1).median()
        result_user["time_step_min"] = (result_user["time_step"]).expanding(
            min_periods=1).min()
        result_user["time_step_max"] = (result_user["time_step"]).expanding(
            min_periods=1).max()
        result_user["time_step_var"] = (result_user["time_step"]).expanding(
            min_periods=1).var()

        result_act = df.drop_duplicates(subset='numberOfActivations', keep='first').set_index(
            ['numberOfActivations'], verify_integrity=True, drop=False).sort_index()
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
        result_act["time_step_mean"] = (result_act["time_step"]).expanding(
            min_periods=1).mean()
        result_act["time_step_cv"] = (result_act["time_step"]).expanding(
            min_periods=1).std()
        result_act["time_step_median"] = (result_act["time_step"]).expanding(
            min_periods=1).median()
        result_act["time_step_min"] = (result_act["time_step"]).expanding(
            min_periods=1).min()
        result_act["time_step_max"] = (result_act["time_step"]).expanding(
            min_periods=1).max()
        result_act["time_step_var"] = (result_act["time_step"]).expanding(
            min_periods=1).var()
        return result_act, result_user


    def liniar_regression(self, df, features = [], target = "" ,test_size = 0.20, random_state = 0):
        X_train, X_test, Y_train, Y_test = sklearn.cross_validation.train_test_split(df[features],
                                                                                     df[target],
                                                                                     test_size=test_size, random_state=random_state)

        lm = LinearRegression(normalize=True)
        lm.fit(X_train, Y_train)

        mse_train = mean_squared_error(Y_train, lm.predict(X_train))
        mse_test = mean_squared_error(Y_test, lm.predict(X_test))

        return mse_train, mse_test


    def steps(self):
        return [MRStep(
            mapper=self.mapper,
            reducer=self.reducer
               )]

if __name__ == '__main__':
    MRJobPopularityRaw.run()
