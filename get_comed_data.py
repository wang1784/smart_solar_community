import pandas as pd
import numpy as np

def get_comed_data():
    # url = 'https://raw.githubusercontent.com/wang1784/smart_solar_community/blame/main/datasets/ComEdAccountsbyPremiseZipCode.csv'
    df = pd.read_csv('ComEdAccountsbyPremiseZipCode.csv')
    df_chicago = df[df['Name'] == 'Chicago, IL']
    df_chicago = df_chicago.astype({'Accounts':int})
    chicago_acct = df_chicago['Accounts']
    print(df_chicago)
    # print(chicago_acct)

get_comed_data()