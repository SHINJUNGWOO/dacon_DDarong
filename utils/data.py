import pandas as pd
import yaml


def remove_nan(dataframe):
    col_names = dataframe.columns

    for col in col_names:
        dataframe[col] = dataframe[col].fillna(0)

    return dataframe


def load_csv(data_pth):
    '''

    :param data_pth: path to csv file
    :return:
        data: pandas dataframe for input data
        target: pandas dataframe for target value, None for test data
    '''
    dataframe = pd.read_csv(data_pth)
    dataframe = remove_nan(dataframe)

    if "rental" in dataframe.columns:
        # for train data
        target = dataframe["rental"]
        data = dataframe.drop(columns ="rental")
    else:
        # for test data
        target = None
        data = dataframe

    return data, target


def load_config(yaml_pth):
    with open(yaml_pth, 'r') as config_file:
        configs = yaml.load(config_file, Loader=yaml.FullLoader)

    return configs
