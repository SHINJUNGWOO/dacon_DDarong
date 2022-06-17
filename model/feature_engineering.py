from pytimekr import pytimekr
from datetime import datetime


def feature_engineering(config, dataframe):
    dataframe = feature_engineering_wrapper(config, dataframe)

    # do extra engineering if needed
    dataframe['dow'] = dataframe['dow'].astype('category')
    dataframe['hol'] = dataframe['hol'].astype('category')

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
