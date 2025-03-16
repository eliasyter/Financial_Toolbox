import yfinance as yf
import numpy as np
import numpy.linalg as npl
import pandas as pd

def Portfolio_Diagnostic(a, w):
    assert np.sum(w)==1
    return None