import json

import pandas as pd


def process_frame(data):
    import matplotlib.pyplot as plt

    result_act = pd.read_json(data["result_act"])
    result_user = pd.read_json(data["result_user"])
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 15))

    usagedominance = result_act[["usagedominance"]]
    usagedominance.columns = ['m1']

    pl = usagedominance.plot(ax=axes[0, 0])
    pl.set_xlabel('')
    # plt.set_ylabel(r'$\frac{r}{r_{M1}}$', rotation=0)

    userusagedominance = result_user[["userusagedominance"]]
    userusagedominance.columns = ['m1']
    # userusagedominance = userusagedominance.apply(rowRatio,1)

    pl = userusagedominance.plot(ax=axes[0, 1])
    pl.set_xlabel('')
    # plt.set_ylabel(r'$\frac{g}{g_{M1}}$', rotation=0)

    usageEntorpy = result_act[["usageEntorpy"]]
    usageEntorpy.columns = ['m1']
    # usageEntorpy = usageEntorpy.apply(rowRatio,1)

    pl = usageEntorpy.plot(ax=axes[1, 0])
    pl.set_xlabel('')
    # # plt.set_ylabel(r'$\frac{H^p}{N_{M1}^p}$', rotation=0)

    userUsageEntorpy = result_user[["userUsageEntorpy"]]
    userUsageEntorpy.columns = ['m1']
    # userUsageEntorpy = userUsageEntorpy.apply(rowRatio,1)

    pl = userUsageEntorpy.plot(ax=axes[1, 1])
    pl.set_xlabel('')
    # # plt.set_ylabel(r'$\frac{H^u}{N_{M1}^u}$', rotation=0)

    ActivateionExposure = result_act[["ActivateionExposure"]]
    ActivateionExposure.columns = ['m1']
    ActivateionExposure.loc[:, 'm1'] = ActivateionExposure.loc[:, 'm1'].rolling(window=5, center=False).mean()

    pl = ActivateionExposure.plot(ax=axes[2, 0])
    pl.set_xlabel('Posts (P)')
    # # plt.set_ylabel(r'$\frac{N^p}{N_{M1}^p}$', rotation=0)

    UserExposure = result_user[["UserExposure"]]
    UserExposure.columns = ['m1']
    UserExposure.loc[:, 'm1'] = UserExposure.loc[:, 'm1'].rolling(window=5, center=False).mean()

    pl = UserExposure.plot(ax=axes[2, 1])
    pl.set_xlabel('Users (U)')
    # # plt.set_ylabel(r'$\frac{N^u}{N_{M1}^u}$', rotation=0)

    plt.savefig("./../images/" + data["name"])


if __name__ == '__main__':

    with open('./../output/geo-ino-track/part-00000') as data_file:
        for frame in data_file:
            process_frame(json.loads(frame))
