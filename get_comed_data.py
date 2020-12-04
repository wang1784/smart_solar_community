import pandas as pd
import numpy as np

def get_comed_data():
    url = 'https://raw.githubusercontent.com/wang1784/smart_solar_community/blame/main/datasets/ComEdAccountsbyPremiseZipCode.csv'
    df = pd.read_csv(url, index_col = 0)
    print(df.shape[0])

get_comed_data()