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
from mrjob.protocol import TextProtocol
from mrjob.step import MRStep
import numpy as np

class toCSV(MRJob):

    INPUT_PROTOCOL = JSONValueProtocol
    OUTPUT_PROTOCOL = TextProtocol

    def mapper(self, key, value):
        df = pd.read_json(value["raw"])
        dfu, df = self.generate_tables(df)

        for index, row in dfu.iterrows():
            v = [index, np.divide(float(index), len(dfu)), value["name"], row["constraint_mean"], row["constraint_var"], len(dfu)]
            yield None, ','.join([str(i) for i in v])

    def steps(self):
        return [MRStep(
            mapper=self.mapper
               )]


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

        result_user["step_distance_mean"] = result_user["step_distance"].expanding(min_periods=1).mean()
        result_user["step_distance_median"] = result_user["step_distance"].expanding(min_periods=1).median()
        result_user["step_distance_cv"] = result_user["step_distance"].expanding(min_periods=1).std()
        result_user["step_distance_var"] = result_user["step_distance"].expanding(min_periods=1).var()
        result_user["step_distance_max"] = result_user["step_distance"].expanding(min_periods=1).max()
        result_user["step_distance_min"] = result_user["step_distance"].expanding(min_periods=1).min()

        result_user["user_exposure_mean"] = result_user["user_exposure"].expanding(min_periods=1).mean()
        result_user["user_exposure_cv"] = result_user["user_exposure"].expanding(min_periods=1).std()
        result_user["user_exposure_var"] = result_user["user_exposure"].expanding(min_periods=1).var()
        result_user["user_exposure_median"] = result_user["user_exposure"].expanding(min_periods=1).median()
        result_user["user_exposure_max"] = result_user["user_exposure"].expanding(min_periods=1).max()
        result_user["user_exposure_min"] = result_user["user_exposure"].expanding(min_periods=1).min()

        result_user["activateion_exposure_mean"] = result_user["activateion_exposure"].expanding(
            min_periods=1).mean()
        result_user["activateion_exposure_cv"] = result_user["activateion_exposure"].expanding(
            min_periods=1).std()
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
        result_user["pagerank_max"] = result_user["pagerank"].expanding(min_periods=1).max()
        result_user["pagerank_min"] = result_user["pagerank"].expanding(min_periods=1).min()

        result_user["constraint_mean"] = result_user["constraint"].expanding(min_periods=1).mean()
        result_user["constraint_cv"] = result_user["constraint"].expanding(min_periods=1).std()
        result_user["constraint_var"] = result_user["constraint"].expanding(min_periods=1).var()
        result_user["constraint_median"] = result_user["constraint"].expanding(min_periods=1).median()
        result_user["constraint_max"] = result_user["constraint"].expanding(min_periods=1).max()
        result_user["constraint_min"] = result_user["constraint"].expanding(min_periods=1).min()

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

        # index on the number of activations
        result_act = df.drop_duplicates(subset='number_activations', keep='first').set_index(
            ['number_activations'], verify_integrity=True, drop=False).sort_index()

        # Surface setup
        result_act["surface_mean"] = result_act["surface"].expanding(min_periods=1).mean()
        result_act["surface_cv"] = result_act["surface"].expanding(min_periods=1).std()
        result_act["surface_var"] = result_act["surface"].expanding(min_periods=1).var()

        # Degre setup
        result_act["degree_mean"] = result_act["degree"].expanding(min_periods=1).mean()
        result_act["degree_median"] = result_act["degree"].expanding(min_periods=1).median()
        result_act["degree_cv"] = result_act["degree"].expanding(min_periods=1).std()
        result_act["degree_var"] = result_act["degree"].expanding(min_periods=1).var()
        result_act["degree_max"] = result_act["degree"].expanding(min_periods=1).max()
        result_act["degree_min"] = result_act["degree"].expanding(min_periods=1).min()

        result_act["step_distance_mean"] = result_act["step_distance"].expanding(min_periods=1).mean()
        result_act["step_distance_median"] = result_act["step_distance"].expanding(min_periods=1).median()
        result_act["step_distance_cv"] = result_act["step_distance"].expanding(min_periods=1).std()
        result_act["step_distance_var"] = result_act["step_distance"].expanding(min_periods=1).var()
        result_act["step_distance_max"] = result_act["step_distance"].expanding(min_periods=1).max()
        result_act["step_distance_min"] = result_act["step_distance"].expanding(min_periods=1).min()

        # Activation exposure setup
        result_act["activateion_exposure_mean"] = result_act["activateion_exposure"].expanding(
            min_periods=1).mean()
        result_act["activateion_exposure_cv"] = result_act["activateion_exposure"].expanding(
            min_periods=1).std()
        result_act["activateion_exposure_var"] = result_act["activateion_exposure"].expanding(
            min_periods=1).var()
        result_act["activateion_exposure_median"] = result_act["activateion_exposure"].expanding(
            min_periods=1).median()
        result_act["activateion_exposure_max"] = result_act["activateion_exposure"].expanding(
            min_periods=1).max()
        result_act["activateion_exposure_min"] = result_act["activateion_exposure"].expanding(
            min_periods=1).min()

        # User exposure setup
        result_act["user_exposure_mean"] = result_act["user_exposure"].expanding(min_periods=1).mean()
        result_act["user_exposure_cv"] = result_act["user_exposure"].expanding(min_periods=1).std()
        result_act["user_exposure_var"] = result_act["user_exposure"].expanding(min_periods=1).var()
        result_act["user_exposure_median"] = result_act["user_exposure"].expanding(min_periods=1).median()
        result_act["user_exposure_max"] = result_act["user_exposure"].expanding(min_periods=1).max()
        result_act["user_exposure_min"] = result_act["user_exposure"].expanding(min_periods=1).min()

        # Pagerank setup
        result_act["pagerank_mean"] = result_act["pagerank"].expanding(min_periods=1).mean()
        result_act["pagerank_cv"] = result_act["pagerank"].expanding(min_periods=1).std()
        result_act["pagerank_var"] = result_act["pagerank"].expanding(min_periods=1).var()
        result_act["pagerank_median"] = result_act["pagerank"].expanding(min_periods=1).median()
        result_act["pagerank_max"] = result_act["pagerank"].expanding(min_periods=1).max()
        result_act["pagerank_min"] = result_act["pagerank"].expanding(min_periods=1).min()

        # constraint setup
        result_act["constraint_mean"] = result_act["constraint"].expanding(min_periods=1).mean()
        result_act["constraint_cv"] = result_act["constraint"].expanding(min_periods=1).std()
        result_act["constraint_var"] = result_act["constraint"].expanding(min_periods=1).var()
        result_act["constraint_median"] = result_act["constraint"].expanding(min_periods=1).median()
        result_act["constraint_max"] = result_act["constraint"].expanding(min_periods=1).max()
        result_act["constraint_min"] = result_act["constraint"].expanding(min_periods=1).min()

        # Time step setup
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


if __name__ == '__main__':
    toCSV.run()
