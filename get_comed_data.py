import pandas as pd
import numpy as np

class get_comed_data():
    def __init__(self):
        self.df = pd.read_csv('ComEdAccountsbyPremiseZipCode.csv', thousands = ',')
        self.chicago = self.df[self.df['Name'] == 'Chicago, IL']
    #
    # def get_acct_int(self):
    #     chicago_acct =
    #     for eachacct in range(len(self.chicago['Accounts'])):
    #         acct_int = eachacct.strip(',')
    #         self.chicago.loc[each]

    def find_chicago_acct(self):
        df_chicago = self.df[self.df['Name'] == 'Chicago, IL']
        df_chicago = df_chicago.astype({'Accounts':int})
        chicago_acct = df_chicago['Accounts']
        print(df_chicago)
    # print(chicago_acct)

get_comed_data.find_chicago_acct()