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
import sklearn
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import KFold
from sklearn.cluster import KMeans

def dt(X):
    return datetime.datetime.fromtimestamp(float(X / 1000))


def to_date(X):
    return X.day()


class MRJobPopularityRaw(MRJob):

    INPUT_PROTOCOL = JSONValueProtocol
    OUTPUT_PROTOCOL = JSONValueProtocol

    #TODO get this so that it is read in from a file
    combinations = {
        "time":["time_step_mean","time_step_cv"],
        "basic":["surface","number_activated_users","number_activations"],
        "community":["inffected_communities_normalised","activation_entorpy","activation_entorpy","usage_dominace","user_usage_dominance"],
        "exposure":["user_exposure_mean", "activateion_exposure_mean"],
        "all":["time_step_mean","time_step_cv","surface","number_activated_users","number_activations","inffected_communities_normalised","activation_entorpy","activation_entorpy","usage_dominace","user_usage_dominance","user_exposure_mean", "activateion_exposure_mean"],
        "time-cluster": ["time_step_mean", "time_step_cv","cluster"],
        "basic-cluster": ["surface", "number_activated_users", "number_activations","cluster"],
        "community-cluster": ["inffected_communities_normalised", "activation_entorpy", "activation_entorpy", "usage_dominace",
                      "user_usage_dominance","cluster"],
        "exposure-cluster": ["user_exposure_mean", "activateion_exposure_mean","cluster"],
        "all-cluster": ["time_step_mean", "time_step_cv", "surface", "number_activated_users", "number_activations",
                "inffected_communities_normalised", "activation_entorpy", "activation_entorpy", "usage_dominace",
                "user_usage_dominance", "user_exposure_mean", "activateion_exposure_mean","cluster"]

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

        df["user_target"] = df["number_activated_users"].values[-1]
        df["activation_target"] = df["number_activations"].values[-1]

        for k, v in df.reset_index().iterrows():
                yield k, {"df": v.to_json(),
                                    "word": line["file"].split("/")[-1],
                                    "period": 30,
                                    "popularity": self.compute_popularity(df, k)[0]}

    def compute_popularity(self, df, days):

        up = []
        p = []
        for x in range(0, days+1):
            dft = df[["number_activations", "number_activated_users"]]
            dft = dft.resample('d').max()
            idx = pd.date_range(dft.index[0], dft.index[0] + datetime.timedelta(days=x))
            dft = dft.reindex(idx, fill_value=0, method='ffill').fillna(method='ffill')

            up.append((dft["number_activated_users"] / dft["number_activated_users"][-1]).mean())
            p.append((dft["number_activations"] / dft["number_activations"][-1]).mean())

        return up, p

    def reducer(self, key, values):
        #TODO compute the populaity K-Means class, this will be a cotogory valibal in the linear regression
        df = {}
        df_kmean = {}

        for v in values:
            df[v["word"]] = json.loads(v["df"])
            df_kmean[v["word"]] = v["popularity"]

        df = pd.DataFrame(df).T
        df_kmean = pd.DataFrame(df_kmean).T

        #Learn the cluster mebership upuntill this time
        x_cols = df_kmean.columns
        cluster = KMeans(n_clusters=2)
        df_kmean['cluster'] = cluster.fit_predict(df_kmean[x_cols])

        #join the cluster membership to the other metrics
        df = df.join(df_kmean)
        print df
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
