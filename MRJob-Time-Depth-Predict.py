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
from sklearn.cross_validation import KFold

def dt(X):
    return datetime.datetime.fromtimestamp(float(X / 1000))


def to_date(X):
    return X.day()


class MRJobPopularityRaw(MRJob):

    INPUT_PROTOCOL = JSONValueProtocol
    OUTPUT_PROTOCOL = JSONValueProtocol

    combinations = {
        "time":["time_step_mean","time_step_cv"],
        "basic":["surface","number_activated_users","number_activations"],
        "community":["inffected_communities_normalised","activation_entorpy","activation_entorpy","usage_dominace","user_usage_dominance"],
        "exposure":["user_exposure_mean", "activateion_exposure_mean"],
        "all":["time_step_mean","time_step_cv","surface","number_activated_users","number_activations","inffected_communities_normalised","activation_entorpy","activation_entorpy","usage_dominace","user_usage_dominance","user_exposure_mean", "activateion_exposure_mean"]
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

        df = df.resample('d').mean()
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
                    r = self.liniar_regression(df.fillna(0), features=v, target=t)

                    yield None, {"observation_level": key, "result_mean": r[0],  "result_var": r[1], "combination":k, "target":t}

    def generate_tables(self, df):
        result_user = df.drop_duplicates(subset='number_activated_users', keep='first').set_index(
            ['number_activated_users'], verify_integrity=True, drop=False).sort_index()
        result_user["surface_mean"] = result_user["surface"].expanding(min_periods=1).mean()
        result_user["surface_cv"] = result_user["surface"].expanding(min_periods=1).std()
        result_user["surface_var"] = result_user["surface"].expanding(min_periods=1).var()
        result_user["degree_mean"] = result_user["degree"].expanding(min_periods=1).mean()
        result_user["degree_median"] = result_user["degree"].expanding(min_periods=1).median()
        result_user["degree_cv"] = result_user["degree"].expanding(min_periods=1).std()
        result_user["degree_var"] = result_user["degree"].expanding(min_periods=1).var()
        result_user["degree_max"] = result_user["degree"].expanding(min_periods=1).max()
        result_user["degree_min"] = result_user["degree"].expanding(min_periods=1).min()
        result_user["user_exposure_mean"] = result_user["user_exposure"].expanding(min_periods=1).mean()
        result_user["user_exposure_cv"] = result_user["user_exposure"].expanding(min_periods=1).std()
        result_user["user_exposure_var"] = result_user["user_exposure"].expanding(min_periods=1).var()
        result_user["user_exposure_median"] = result_user["user_exposure"].expanding(min_periods=1).median()
        result_user["user_exposure_min"] = result_user["user_exposure"].expanding(min_periods=1).max()
        result_user["user_exposure_mean"] = result_user["user_exposure"].expanding(min_periods=1).min()
        result_user["activateion_exposure_mean"] = result_user["activateion_exposure"].expanding(
            min_periods=1).mean()
        result_user["activateion_exposure_cv"] = result_user["activateion_exposure"].expanding(
            min_periods=1).std()
        result_user["activateion_exposure_var"] = result_user["activateion_exposure"].expanding(
            min_periods=1).var()
        result_user["activateion_exposure_var"] = result_user["activateion_exposure"].expanding(
            min_periods=1).var()
        result_user["activateion_exposure_median"] = result_user["activateion_exposure"].expanding(
            min_periods=1).median()
        result_user["activateion_exposure_max"] = result_user["activateion_exposure"].expanding(
            min_periods=1).max()
        result_user["activateion_exposure_min"] = result_user["activateion_exposure"].expanding(
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

        result_act = df.drop_duplicates(subset='number_activations', keep='first').set_index(
            ['number_activations'], verify_integrity=True, drop=False).sort_index()
        result_act["surface_mean"] = result_act["surface"].expanding(min_periods=1).mean()
        result_act["surface_cv"] = result_act["surface"].expanding(min_periods=1).std()
        result_act["surface_var"] = result_act["surface"].expanding(min_periods=1).var()
        result_act["degree_mean"] = result_act["degree"].expanding(min_periods=1).mean()
        result_act["degree_median"] = result_act["degree"].expanding(min_periods=1).median()
        result_act["degree_cv"] = result_act["degree"].expanding(min_periods=1).std()
        result_act["degree_var"] = result_act["degree"].expanding(min_periods=1).var()
        result_act["degree_max"] = result_act["degree"].expanding(min_periods=1).max()
        result_act["degree_min"] = result_act["degree"].expanding(min_periods=1).min()
        result_act["activateion_exposure_mean"] = result_act["activateion_exposure"].expanding(
            min_periods=1).mean()
        result_act["activateion_exposure_cv"] = result_act["activateion_exposure"].expanding(
            min_periods=1).std()
        result_act["activateion_exposure_var"] = result_act["activateion_exposure"].expanding(
            min_periods=1).var()
        result_act["activateion_exposure_var"] = result_act["activateion_exposure"].expanding(
            min_periods=1).var()
        result_act["activateion_exposure_median"] = result_act["activateion_exposure"].expanding(
            min_periods=1).median()
        result_act["activateion_exposure_max"] = result_act["activateion_exposure"].expanding(
            min_periods=1).max()
        result_act["activateion_exposure_min"] = result_act["activateion_exposure"].expanding(
            min_periods=1).min()
        result_act["user_exposure_mean"] = result_act["user_exposure"].expanding(min_periods=1).mean()
        result_act["user_exposure_cv"] = result_act["user_exposure"].expanding(min_periods=1).std()
        result_act["user_exposure_var"] = result_act["user_exposure"].expanding(min_periods=1).var()
        result_act["user_exposure_median"] = result_act["user_exposure"].expanding(min_periods=1).median()
        result_act["user_exposure_min"] = result_act["user_exposure"].expanding(min_periods=1).max()
        result_act["user_exposure_mean"] = result_act["user_exposure"].expanding(min_periods=1).min()
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


    def liniar_regression(self, df, features = [], target = "" , nfolds = 15):
        kf = KFold(len(df), n_folds=nfolds, shuffle=True)
        results_array = []
        for train_index, test_index in kf:
            X_train, X_test = df.ix[train_index, features], df.ix[test_index, features]
            Y_train, Y_test = df.ix[train_index, target], df.ix[test_index, target]

            lm = LinearRegression(normalize=True)
            lm.fit(X_train, Y_train)

            results_array.append(mean_squared_error(Y_test, lm.predict(X_test)))

        return np.mean(results_array), np.var(results_array)


    def steps(self):
        return [MRStep(
            mapper=self.mapper,
            reducer=self.reducer
               )]

if __name__ == '__main__':
    MRJobPopularityRaw.run()
