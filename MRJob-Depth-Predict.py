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

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error,accuracy_score
from sklearn.cross_validation import KFold
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import StratifiedKFold

def dt(X):
    return datetime.datetime.fromtimestamp(float(X / 1000))


def to_date(X):
    return X.day()


class MRJobPopularityRaw(MRJob):

    INPUT_PROTOCOL = JSONValueProtocol
    OUTPUT_PROTOCOL = JSONValueProtocol

    combinations = {
        "time": ["time_step_mean", "time_step_cv", "early_spread_time"],
        "basic": ["surface", "number_activated_users", "number_activations"],
        "community": ["inffected_communities_normalised", "activation_entorpy", "user_usage_entorpy", "usage_dominace",
                      "user_usage_dominance"],
        "exposure": ["user_exposure_mean", "activateion_exposure_mean"],
        "cascades": ["wiener_index_avrage", "wiener_index_std", "number_of_trees", "cascade_edges", "cascade_nodes"],
        "distance": ["diamiter"],
        "broker": ["gatekeeper", "liaison", "representative", "coordinator", "consultant"],
        "all": ["time_step_mean", "time_step_cv", "early_spread_time", "surface", "number_activated_users",
                "number_activations", "inffected_communities_normalised", "activation_entorpy", "user_usage_entorpy",
                "usage_dominace", "user_usage_dominance", "user_exposure_mean", "activateion_exposure_mean",
                "wiener_index_avrage", "wiener_index_std", "number_of_trees", "cascade_edges", "cascade_nodes",
                "diamiter", "gatekeeper", "liaison", "representative", "coordinator", "consultant"]
        # "all":["time_step_mean","time_step_cv","surface","number_activated_users","number_activations","inffected_communities_normalised","activation_entorpy","activation_entorpy","usage_dominace","user_usage_dominance","user_exposure_mean", "activateion_exposure_mean","wiener_index_avrage","number_of_trees"]
        # "all":["time_step_mean","time_step_cv","surface","number_activated_users","number_activations","inffected_communities_normalised","activation_entorpy","activation_entorpy","usage_dominace","user_usage_dominance","user_exposure_mean", "activateion_exposure_mean"]
    }

    target = ["popularity_class","user_popularity_class"]

    def configure_options(self):
        super(MRJobPopularityRaw, self).configure_options()
        self.add_passthrough_option('--avrage', type='int', default=0, help='...')
        self.add_passthrough_option('--cluster', type='int', default=10, help='...')
        self.add_passthrough_option('--folds', type='int', default=10, help='...')
        self.add_passthrough_option('--classifier', type='string', default="logit_regression", help='...')
        self.add_passthrough_option('--day_from', type='int', default=15, help='...')
        self.add_passthrough_option('--day_to', type='int', default=45, help='...')

    def mapper(self, _, line):
        df = pd.read_json(line["raw"])
        dfu, df = self.generate_tables(df)

        df['time'] = df['time'].apply(dt)
        df = df.set_index(pd.DatetimeIndex(df['time']))
        dfu['time'] = dfu['time'].apply(dt)
        dfu = dfu.set_index(pd.DatetimeIndex(dfu['time']))


        df["user_target"] = df["number_activated_users"].values[-1]
        df["activation_target"] = df["number_activations"].values[-1]

        for k, v in df.reset_index().iterrows():
                yield {"observations":k, "type":["popularity_class"]}, {"df": v.to_json(),
                                    "word": line["file"].split("/")[-1]}


        dfu["user_target"] = dfu["number_activated_users"].values[-1]
        dfu["activation_target"] = dfu["number_activations"].values[-1]

        for k, v in dfu.reset_index().iterrows():
                yield {"observations":k, "type":["user_popularity_class"]}, {"df": v.to_json(),
                                    "word": line["file"].split("/")[-1]}

    def mapper_time(self, _, line):
        df = pd.read_json(line["raw"])
        dfu, df = self.generate_tables(df)

        df['time'] = df['time'].apply(dt)
        df = df.set_index(pd.DatetimeIndex(df['time']))

        dfu['time'] = dfu['time'].apply(dt)
        dfu = dfu.set_index(pd.DatetimeIndex(dfu['time']))

        dfre = df.resample('d').mean()
        idx = pd.date_range(dfre.index[0], dfre.index[0] + datetime.timedelta(days=self.options.day_to))
        dfi = dfre.reindex(idx, fill_value=0, method='ffill').fillna(method='ffill')

        for k, v in df.reset_index().iterrows():
            for kt in range(self.options.day_from, self.options.day_to):
                # dft = dfi[:kt]

                v["user_target"] = dfi["number_activated_users"].values[kt]
                v["activation_target"] = dfi["number_activations"].values[kt]

                yield {"observations":k, "type":["popularity_class"], "period":kt}, {"df": v.to_json(),
                                    "word": line["file"].split("/")[-1]}


        dfure = dfu.resample('d').mean()
        idx = pd.date_range(dfure.index[0], dfure.index[0] + datetime.timedelta(days=self.options.day_to))
        dfi = dfure.reindex(idx, fill_value=0, method='ffill').fillna(method='ffill')

        for k, v in dfu.reset_index().iterrows():
            for kt in range(self.options.day_from, self.options.day_to):
                # dft = dfi[:kt]

                v["user_target"] = dfi["number_activated_users"].values[kt]
                v["activation_target"] = dfi["number_activations"].values[kt]

                yield {"observations":k, "type":["user_popularity_class"], "period":kt}, {"df": v.to_json(),
                                        "word": line["file"].split("/")[-1]}


    def classifier(self, X, avr):
        if X >= avr:
            return True
        else:
            return False

    def reducer_logit_time(self, key, values):

        df = {}
        for v in values:
            df[v["word"]] = json.loads(v["df"])
        df = pd.DataFrame(df).T.fillna(0)

        avr_pop = df["activation_target"].mean()
        avr_user_pop = df["user_target"].mean()
        #Todo use this in a map to create a new colum which indicates that
        # it is either above or below the avrage target, for each
        # the two colums would be
        df["popularity_class"] = df["activation_target"].apply(self.classifier, args=(avr_pop,))
        df["user_popularity_class"] = df["user_target"].apply(self.classifier, args=(avr_user_pop,))

        if len(df) > 1:
            for t in key["type"]:
                if self.options.folds > len(df[t].values):
                    f = len(df[t].values)
                else:
                    f = self.options.folds
                kf = StratifiedKFold(df[t].values, n_folds=f, shuffle=True)
                for train_index, test_index in kf:
                    for k, v in self.combinations.iteritems():
                        X_train, X_test = df.ix[train_index, v], df.ix[test_index, v]
                        Y_train, Y_test = df.ix[train_index, t], df.ix[test_index, t]
                        lm = LogisticRegression()
                        if len(set(Y_train)) > 1 and len(X_train) > 1 and len(X_test) > 1 and len(set(Y_test)) > 1:
                            lm.fit(X_train, Y_train)
                            r = accuracy_score(Y_test, lm.predict(X_test))
                            yield None, {"observation_level": key["observations"], "result": r, "combination":k, "target":t, "conf":lm.coef_.tolist(), "time":key["period"]}

    def reducer_logit(self, key, values):

        df = {}
        for v in values:
            df[v["word"]] = json.loads(v["df"])
        df = pd.DataFrame(df).T.fillna(0)

        avr_pop = df["activation_target"].mean()
        avr_user_pop = df["user_target"].mean()
        #Todo use this in a map to create a new colum which indicates that
        # it is either above or below the avrage target, for each
        # the two colums would be
        df["popularity_class"] = df["activation_target"].apply(self.classifier, args=(avr_pop,))
        df["user_popularity_class"] = df["user_target"].apply(self.classifier, args=(avr_user_pop,))

        if len(df) > 1:
            for t in key["type"]:
                if self.options.folds > len(df[t].values):
                    f = len(df[t].values)
                else:
                    f = self.options.folds
                kf = StratifiedKFold(df[t].values, n_folds=f, shuffle=True)
                for train_index, test_index in kf:
                    for k, v in self.combinations.iteritems():
                        X_train, X_test = df.ix[train_index, v], df.ix[test_index, v]
                        Y_train, Y_test = df.ix[train_index, t], df.ix[test_index, t]
                        lm = LogisticRegression()
                        if len(set(Y_train)) > 1 and len(X_train) > 1 and len(X_test) > 1 and len(set(Y_test)) > 1:
                            lm.fit(X_train, Y_train)
                            r = accuracy_score(Y_test, lm.predict(X_test))
                            yield None, {"observation_level": key["observations"], "result": r, "combination":k, "target":t, "conf":lm.coef_.tolist()}

    def reducer_liniar(self, key, values):
        #TODO compute the populaity K-Means class, this will be a cotogory valibal in the linear regression
        df = {}

        for v in values:
            df[v["word"]] = json.loads(v["df"])

        df = pd.DataFrame(df).T.fillna(0)

        if len(df) > 1:

            #Generate the kfolds
            kf = KFold(len(df), n_folds=self.options.folds, shuffle=True)
            for train_index, test_index in kf:


                for t in self.target:
                    for k, v in self.combinations_no_c.iteritems():

                        #Generate the test and train datsets
                        X_train, X_test = df.ix[train_index, v], df.ix[test_index, v]
                        Y_train, Y_test = df.ix[train_index, t], df.ix[test_index, t]

                        lm = LinearRegression(normalize=True)
                        lm.fit(X_train, Y_train)
                        r = mean_squared_error(Y_test, lm.predict(X_test))
                        # yield None, {"observation_level": key["observations"], "result": r, "combination":k, "target":t, "target_level": key["target"],"clusters":cnum, "cluster_num":int(num), "popmessure":popk, "conf":lm.coef_.tolist()}



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

    def steps(self):
        # return [MRStep(
        #     mapper=self.mapper,
        #     reducer=self.reducer_logit
        #        )]

        return [MRStep(
            mapper=self.mapper_time,
            reducer=self.reducer_logit_time
        )]

if __name__ == '__main__':
    MRJobPopularityRaw.run()
