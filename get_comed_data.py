import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

###use chicago_hourly function to get a dataframe of hourly consumption per account

class get_comed_data():
    def __init__(self):
        #read dataframe
        self.zip = pd.read_csv('ComEdAccountsbyPremiseZipCode.csv', thousands = ',')
        self.chicago = self.zip[self.zip['Name'] == 'Chicago, IL']
        self.zip = self.zip.astype({'Accounts':int}) #make number of accounts integer
        self.chicago = self.chicago.astype({'Accounts': int}) #make number of accounts integer
        self.comed = pd.read_csv('COMED_hourly.csv')

    def find_chicago_acct(self):
        #find number of chicago accounts
        chicago_acct = sum(self.chicago['Accounts'])
        return chicago_acct

    def total_acct(self):
        #find total number of accounts in COMED
        accts = sum(self.zip['Accounts'])
        return accts

    def chicago_perc(self):
        #find the percentage of accounts that are in Chicago area
        chicago_acct = self.find_chicago_acct()
        accts = self.total_acct()
        chicago_percentage = chicago_acct/accts
        return chicago_percentage

    def account_hourly(self):
        #calculate the energy consumption for each account
        total_acct = self.total_acct()
        self.comed['COMED_MW'] = self.comed['COMED_MW'] / total_acct * 1000000
        self.comed.rename(columns = {'COMED_MW':'COMED_W'}, inplace = True)
        self.comed.set_index('Datetime', inplace = True)
        return self.comed


