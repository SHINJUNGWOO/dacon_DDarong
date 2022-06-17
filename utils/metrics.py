import numpy as np


def nmae(true , pred ):
    score = np.mean((np.abs(true-pred))/true)

    return score