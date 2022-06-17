from lightgbm import LGBMRegressor


def build_regressor(config):
    regressor_type = config["type"]
    params = config["params"]

    if regressor_type == "lgbm":
        regressor = LGBMRegressor(**params)

    return regressor