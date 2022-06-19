import os

from pytimekr import pytimekr
from datetime import datetime
from utils.data import load_csv


def feature_engineering(config, dataframe):
    dataframe = corona(dataframe, config)
    dataframe = feature_engineering_wrapper(config, dataframe)
    print(max(dataframe["daily_infection"]))

    # do extra engineering if needed
    dataframe["dow"] = dataframe["dow"].astype("category")
    dataframe["hol"] = dataframe["hol"].astype("category")

    return dataframe


def feature_engineering_wrapper(config, dataframe):
    '''

    :param config: dict that contains feature engineering methods
    :param dataframe: dataset in dataframe
    :return: a new dataframe after applying feature engineering
    '''
    funcs = config["feature_engineering"]

    for func in funcs:
        dataframe = globals()[func](dataframe)

    return dataframe


def date_to_int(dataframe):
    def transform(item):
        item = item.split("-")
        # val = int(item[1])
        val = int(item[0]) * 24 * 60 + int(item[1]) * 31 + int(item[2])
        return val
    dataframe["date"] = dataframe["date"].apply(lambda x: transform(x))
    return dataframe


def finddow(dataframe):
    # returns if a day is weekday or not

    def transform(item):
        return datetime.strptime(item.split()[0],"%Y-%m-%d").weekday()
    dataframe["dow"] = dataframe["date"].apply(lambda x: transform(x))
    return dataframe


def find_hol(dataframe):
    # returns if a day is a holiday or not

    def transform(item):
        year_month_day = item.split("-")
        for i in pytimekr.holidays(int(year_month_day[0])):
            if item == str(i):
                return 1
        else:
            return 0
    dataframe["hol"] = dataframe["date"].apply(lambda x: transform(x))
    dataframe["hol"] = dataframe[['hol', 'dow']].apply(
        lambda x: 1 if (x[0] == 1) or (x[1] in [5, 6]) else 0, axis=1
    )

    return dataframe


def corona(dataframe, config):
    # returns daily corona infections
    # http://data.seoul.go.kr/dataList/OA-20461/S/1/datasetView.do

    # define some subfunctions for covid data parsing
    def to_datetime(item):
        return item.replace('.', '-')[:-3]

    data_pth = config["main"]["data_pth"]
    corona_data, _ = load_csv(os.path.join(data_pth, "covid.csv"))
    corona_data["date"] = corona_data["date"].apply(lambda x: to_datetime(x))

    def transform(corona_data, item):
        if item in list(corona_data["date"]):
            return_idx = list(corona_data["date"]).index(item)
            return corona_data["daily_infection"][return_idx]
        else:
            return 0.0

    dataframe["daily_infection"] = dataframe["date"].apply(
        lambda x: transform(corona_data, x)
    )

    return dataframe
