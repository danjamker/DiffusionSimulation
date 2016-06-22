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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import KFold
from sklearn.cluster import KMeans
import numpy as np
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


    combinations_no_c = {
        "time":["time_step_mean","time_step_cv","early_spread_time"],
        "basic":["surface","number_activated_users","number_activations"],
        "community":["inffected_communities_normalised","activation_entorpy","user_usage_entorpy","usage_dominace","user_usage_dominance"],
        "exposure":["user_exposure_mean", "activateion_exposure_mean"],
        "cascades":["wiener_index_avrage","wiener_index_std","number_of_trees","cascade_edges","cascade_nodes"],
        "distance":["diamiter"],
        "broker":["gatekeeper","liaison","representative","coordinator","consultant"],
        "all":["time_step_mean","time_step_cv","early_spread_time","surface","number_activated_users","number_activations","inffected_communities_normalised","activation_entorpy","user_usage_entorpy","usage_dominace","user_usage_dominance","user_exposure_mean", "activateion_exposure_mean","wiener_index_avrage","wiener_index_std","number_of_trees","cascade_edges","cascade_nodes","diamiter","gatekeeper","liaison","representative","coordinator","consultant"]
        # "all":["time_step_mean","time_step_cv","surface","number_activated_users","number_activations","inffected_communities_normalised","activation_entorpy","activation_entorpy","usage_dominace","user_usage_dominance","user_exposure_mean", "activateion_exposure_mean","wiener_index_avrage","number_of_trees"]
        # "all":["time_step_mean","time_step_cv","surface","number_activated_users","number_activations","inffected_communities_normalised","activation_entorpy","activation_entorpy","usage_dominace","user_usage_dominance","user_exposure_mean", "activateion_exposure_mean"]
    }

    target = ["user_target","activation_target"]

    def configure_options(self):
        super(MRJobPopularityRaw, self).configure_options()
        self.add_passthrough_option('--avrage', type='int', default=0, help='...')
        self.add_passthrough_option('--cluster', type='int', default=5, help='...')
        self.add_passthrough_option('--folds', type='int', default=10, help='...')
        self.add_passthrough_option('--day_from', type='int', default=15, help='...')
        self.add_passthrough_option('--day_to', type='int', default=45, help='...')



    def mapper(self, _, line):
        df = pd.read_json(line["raw"])
        dfu, df = self.generate_tables(df)

        df['time'] = df['time'].apply(dt)
        df = df.set_index(pd.DatetimeIndex(df['time']))

        df = df.resample('d').mean()
        idx = pd.date_range(df.index[0], df.index[0] + datetime.timedelta(days=self.options.day_to))
        dfi = df.reindex(idx, fill_value=0, method='ffill').fillna(method='ffill')

        dfi["user_pop"] = dfi["number_activated_users"].expanding(min_periods=1).apply(self.apply_pop)
        dfi["activation_pop"] = dfi["number_activations"].expanding(min_periods=1).apply(self.apply_pop)

        for kt in range(self.options.day_from, self.options.day_to):

            dft = dfi[:kt]

            dft["user_target"] = dfi["number_activated_users"].values[-1]
            dft["activation_target"] = dfi["number_activations"].values[-1]


            for k, v in dft.reset_index().iterrows():
                if k > 0:
                    # pop = self.compute_popularity(dft, k)
                    yield {"observations":k, "target":kt}, {"df": v.to_json(),
                                        "word": line["file"].split("/")[-1],
                                        "period": kt,
                                        "popularity": v["activation_pop"],
                                        "user_popularity": v["user_pop"]}

    def compute_popularity(self, df, days, resample_granularity = 'd'):
        #TODO could this be changed into an expanding apply
        up = []
        p = []
        dft = df[["number_activations", "number_activated_users"]]
        # dft = dft.resample(resample_granularity).max()
        idx = pd.date_range(dft.index[0], dft.index[0] + datetime.timedelta(days=days))
        dft = dft[:days]

        for x in range(1, days+1):
            up.append((dft[:x]["number_activated_users"] / dft[:x]["number_activated_users"][-1]).mean())
            p.append((dft[:x]["number_activations"] / dft[:x]["number_activations"][-1]).mean())

        return up, p

    def apply_pop(self, X):
        return np.divide(X, X[-1]).mean()

    def reducer(self, key, values):
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
        # print df
        if len(df) > 1:
            for k, v in self.combinations.iteritems():
                for t in self.target:
                    r = self.liniar_regression(df.fillna(0), features=v, target=t)
                    yield None, {"observation_level": key["observations"], "result_mean": r[0],  "result_var": r[1], "combination":k, "target":t, "target-day":key["target"]}

    def reducer_kmean(self, key, values):
        #TODO compute the populaity K-Means class, this will be a cotogory valibal in the linear regression
        df = {}
        df_kmean = {}
        df_kmean_user = {}

        for v in values:
            df[v["word"]] = json.loads(v["df"])
            df_kmean[v["word"]] = v["popularity"]
            df_kmean_user[v["word"]] = v["user_popularity"]

        df = pd.DataFrame(df).T.fillna(0)
        df_kmean = pd.DataFrame(df_kmean).T.fillna(0)
        df_kmean_user = pd.DataFrame(df_kmean_user).T.fillna(0)
        popdict = {"frequency":df_kmean,"user":df_kmean_user}
        #Learn the cluster mebership upuntill this time
        x_cols = df_kmean.columns
        #join the cluster membership to the other metrics
        # print df
        if len(df) > 1:

            #Generate the kfolds
            kf = KFold(len(df), n_folds=self.options.folds, shuffle=True)

            #which popularity kmeans to use
            for popk, popv in popdict.iteritems():

                #iterate though the indecides to test and train
                for train_index, test_index in kf:
                    # Get the K-means test and train data
                    train_kmean = popv.ix[train_index, x_cols]
                    test_kmean = popv.ix[test_index, x_cols]
                    for cnum in range(2, self.options.cluster):

                        cluster = KMeans(n_clusters=cnum)
                        train_kmean['cluster'] = cluster.fit_predict(train_kmean)
                        test_kmean['cluster'] = cluster.predict(test_kmean)

                        test_kmean.fillna(0)
                        train_kmean.fillna(0)

                        for t in self.target:
                            for k, v in self.combinations_no_c.iteritems():

                                #Generate the test and train datsets
                                X_train, X_test = df.ix[train_index, v], df.ix[test_index, v]
                                Y_train, Y_test = df.ix[train_index, t], df.ix[test_index, t]

                                for num in set(test_kmean['cluster'].values):
                                    wor_train = train_kmean[(train_kmean["cluster"] == num)]
                                    wor_test = test_kmean[(test_kmean["cluster"] == num)]
                                    lm = LinearRegression(normalize=True)
                                    if len(Y_train[(Y_train.index.isin(wor_train.index.values))]) > 0 and len(X_train[(X_train.index.isin(wor_train.index.values))]) > 0:

                                        lm.fit(X_train[(X_train.index.isin(wor_train.index.values))], Y_train[(Y_train.index.isin(wor_train.index.values))])
                                        r = mean_squared_error(Y_test[(Y_test.index.isin(wor_test.index.values))], lm.predict(X_test[(X_test.index.isin(wor_test.index.values))]))
                                        yield None, {"observation_level": key["observations"], "result": r, "combination":k, "target":t, "target_level": key["target"],"clusters":cnum, "cluster_num":int(num), "popmessure":popk, "conf":lm.coef_.tolist()}





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


        #index on the number of activations
        result_act = df.drop_duplicates(subset='number_activations', keep='first').set_index(
            ['number_activations'], verify_integrity=True, drop=False).sort_index()

        #Surface setup
        result_act["surface_mean"] = result_act["surface"].expanding(min_periods=1).mean()
        result_act["surface_cv"] = result_act["surface"].expanding(min_periods=1).std()
        result_act["surface_var"] = result_act["surface"].expanding(min_periods=1).var()

        #Degre setup
        result_act["degree_mean"] = result_act["degree"].expanding(min_periods=1).mean()
        result_act["degree_median"] = result_act["degree"].expanding(min_periods=1).median()
        result_act["degree_cv"] = result_act["degree"].expanding(min_periods=1).std()
        result_act["degree_var"] = result_act["degree"].expanding(min_periods=1).var()
        result_act["degree_max"] = result_act["degree"].expanding(min_periods=1).max()
        result_act["degree_min"] = result_act["degree"].expanding(min_periods=1).min()

        #Activation exposure setup
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

        #User exposure setup
        result_act["user_exposure_mean"] = result_act["user_exposure"].expanding(min_periods=1).mean()
        result_act["user_exposure_cv"] = result_act["user_exposure"].expanding(min_periods=1).std()
        result_act["user_exposure_var"] = result_act["user_exposure"].expanding(min_periods=1).var()
        result_act["user_exposure_median"] = result_act["user_exposure"].expanding(min_periods=1).median()
        result_act["user_exposure_min"] = result_act["user_exposure"].expanding(min_periods=1).max()
        result_act["user_exposure_mean"] = result_act["user_exposure"].expanding(min_periods=1).min()

        #Pagerank setup
        result_act["pagerank_mean"] = result_act["pagerank"].expanding(min_periods=1).mean()
        result_act["pagerank_cv"] = result_act["pagerank"].expanding(min_periods=1).std()
        result_act["pagerank_var"] = result_act["pagerank"].expanding(min_periods=1).var()
        result_act["pagerank_var"] = result_act["pagerank"].expanding(min_periods=1).var()
        result_act["pagerank_median"] = result_act["pagerank"].expanding(min_periods=1).median()
        result_act["pagerank_max"] = result_act["pagerank"].expanding(min_periods=1).max()
        result_act["pagerank_min"] = result_act["pagerank"].expanding(min_periods=1).min()

        #Time step setup
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


    def liniar_regression(self, df, features = [], target = "" , nfolds = 15, scoring="mean_squared_error"):
        kf = KFold(len(df), n_folds=nfolds, shuffle=True)
        lm = LinearRegression(normalize=True)
        scores = sklearn.cross_validation.cross_val_score(lm, df[features], df[target], scoring=scoring, cv=kf )
        return scores.mean(), scores.var()

    def steps(self):
        return [MRStep(
            mapper=self.mapper,
            reducer=self.reducer_kmean
               )]

if __name__ == '__main__':
    MRJobPopularityRaw.run()
