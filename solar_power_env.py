from get_comed_data import get_comed_data
import pandas as pd
# from get_solar_data import get_solar_data
#
def combine_data():
    df_comed = get_comed_data().account_hourly()
    df_solar = pd.read_csv('solar_data.csv', names = ['Datetime', 'SOLAR_W']).drop(0)
    df_solar['Datetime'] = df_solar['Datetime'].str.replace('-06:00', '')
    df_solar.set_index('Datetime', inplace = True)
    # df_solar = get_solar_data()
    df_join = df_comed.join(df_solar, how = 'inner', sort = True)
    # print(df_comed.shape, df_solar.shape)
    # print(df_join.shape)
    print(df_join.head())

class solar_power_env():
    def __init__(self, df_join):
        self._data = df_join

    def get_reward(self, state, action):
        #action is either 0 (discharging the battery) or 1 (charging the battery)
        if action:
            reward =