from get_comed_data import get_comed_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from get_solar_data import get_solar_data

### SUPPORT FUNCTIONS ###
def combine_data():
    #get data from comed and solar
    df_comed = get_comed_data().account_hourly() #call function to get comed data
    df_solar = pd.read_csv('solar_data.csv', names = ['Datetime', 'SOLAR_W']).drop(0) #read solar data from csv file
    df_solar['Datetime'] = df_solar['Datetime'].str.replace('-06:00', '') #drop unuseful information so that datetime is consistent with comed data
    df_solar.set_index('Datetime', inplace = True) # set datetime as index
    df_solar[df_solar<0] = 0 #make all the negative values 0 because negative solar generation is meaningless
    df_join = df_comed.join(df_solar, how = 'inner', sort = True) #join two dataframes according to date and time
    df_join['COMED_W'] = df_join['COMED_W']

    #discretize data
    df_join['solar_bin'] = pd.cut(x=df_join['SOLAR_W'],
                             bins=[-0.75,0,50,100,150,200],
                             labels=[0, 1, 2, 3, 4])
    df_join['SOLAR_W'] = df_join['SOLAR_W'] * 16 #assume each household has 16 panels
    df_join['comed_bin'] = pd.cut(x = df_join['COMED_W'],
                                  bins = np.array([1900, 2800, 3100, 4500, 6500]),
                                  labels = [0, 1, 2, 3])
    #print dataframe info
    # print(df_join.iloc[7:21])
    # print(df_join.describe())
    # print('comed\n', df_join['comed_bin'].value_counts())
    # print('solar\n', df_join['solar_bin'].value_counts())
    return df_join

def plot_df_join(df_join):
    #graph the entire dataset
    plt.plot(range(df_join.shape[0]), df_join['COMED_W'], label = 'Load')
    plt.plot(range(df_join.shape[0]), df_join['SOLAR_W'], alpha = 0.5, label = 'Solar')
    plt.xlabel('Hours')
    plt.ylabel('Watts')
    plt.legend()
    plt.title('Power consumed and collected from 20 solar panels 2011 - 2018')
    plt.show()

    #graph the first year
    plt.plot(range(8760), df_join.iloc[:8760]['COMED_W'], label = 'Load')
    plt.plot(range(8760), df_join.iloc[:8760]['SOLAR_W'], alpha = 0.5, label = 'Solar')
    plt.xlabel('Hours')
    plt.ylabel('Watts')
    plt.legend()
    plt.title('Power consumed and collected from 20 solar panels 2011')
    plt.show()

def split_year():
    #split the given df into sub-dfs according to year
    df = combine_data()
    df_index = pd.Series(df.index).str.strip()
    year = []
    for eachindex in df_index:
        year.append(eachindex[:4])
    # data_datetime = pd.to_datetime(df.index, format = '%Y%m%d %H%M%S')
    # data_year = data_datetime.dt.years
    df_temp = df.copy()
    df_temp['Year'] = year
    agg = df_temp.groupby(['Year'])
    year_dfs = []
    start_end = []
    for year, group in agg:
        if group.shape[0]>1:
            year_dfs.append(group)
            start_end.append([group.iloc[0, 0], group.iloc[-1,0]])

    #find years that has smooth transitions in COMED_W
    year_close = {}
    for eachyear in range(len(start_end)):
        year_close[eachyear] = []
        for eachother in range(len(start_end)):
            if eachyear != eachother and abs(start_end[eachyear][1] - start_end[eachother][0]) < 150:
                year_close[eachyear].append(eachother)
    return year_dfs, year_close

def split_month():
    df = combine_data()
    df_index = pd.Series(df.index).str.strip()
    month = []
    year = []
    for eachindex in df_index:
        month.append(eachindex[5:7])
        year.append(eachindex[:4])
    df_temp = df.copy()
    df_temp['Month'] = month
    df_temp['Year'] = year
    agg_month = df_temp.groupby(['Month'])
    month_dfs = []
    for month, month_group in agg_month:
        month_year = []
        agg_year = month_group.groupby(['Year'])
        for year, year_group in agg_year:
            if year_group.shape[0]>1:
                month_year.append(year_group)
        month_dfs.append(month_year)
    return month_dfs


### ENVIRONMENT ###
class solar_power_env():
    def __init__(self):
        #get dataframe
        # self._data = self.generate_episode()
        self._data = self.generate_episode()
        # self._data = pd.concat([temp, temp])

        #battery
        self._battery_cap = 2000 #max power can be stored in the battery
        self._battery_bin = list(np.linspace(0, self._battery_cap, 6)) #make 5 bins for battery level
        self._battery_power = 0 #records the exact power that the battery has

        #power data
        self._solar_bin = [-0.75,0,50,100,150,200] #bins for solar
        self._comed_bin = np.array([1900, 2800, 3100, 4500, 6500]) #bins for
        self._data_step = 0 #row index to extract data from dataframe, starting with index 0
        p_init = self._data.iloc[0] #start with first line
        self._state = (p_init['solar_bin'], p_init['comed_bin'], 0) #initiate state

        #state space shape
        self._state_space_shape = [len(self._solar_bin)-1, len(self._comed_bin)-1, len(self._battery_bin)-1]

    def choose_month(self, df_month):
        num_row = len(df_month)
        choose_month_year = np.random.choice(range(num_row), 1)[0]
        return df_month[choose_month_year]

    def generate_year(self, month_dfs):
        df_year_choices = []
        for eachmonth in range(len(month_dfs)):
            month_year_dfs = month_dfs[eachmonth]
            month_choice = self.choose_month(month_year_dfs)
            df_year_choices.append(month_choice)
            # print(month_choice.shape)
        df_year = pd.concat(df_year_choices)
        return df_year

    def generate_episode(self):
        month_dfs = split_month()
        df_episode = []
        for eachyear in range(15):
            df_each_year = self.generate_year(month_dfs)
            df_episode.append(df_each_year)
        df_episode = pd.concat(df_episode)
        # print(df_episode.shape)
        # num_year = len(year_dfs)
        # year_choose = np.random.choice(range(num_year), 3)
        # print(year_choose)
        # df_selected = [combine_data()]
        # year_choose = [0, 1, 2, 3, 4, 5, 1, 4, 2, 3, 4, 5, 1, 2, 3, 4, 5, 6]
        # df_selected = []
        # for eachchosen in year_choose:
        #     df_selected.append(year_dfs[eachchosen])
        #     # print(year_dfs[eachchosen].shape[0])
        # df_combined = pd.concat(df_selected)
        return df_episode

    #assemble state information
    def extract_from_table(self, item): #finds p_solar, p_comed, OR bin_solar, bin_comed
        #item: indicate whether if it's 'power' or 'bin' data that we want to extract from dataframe
        p = self._data.iloc[self._data_step] #locate the row that we want to extract information from
        if item == 'power':
            return p['SOLAR_W'], p['COMED_W']
        elif item == 'bin':
            return p['solar_bin'], p['comed_bin']

    def battery_level(self, change): #finds battery level with given change in power stored in battery
        #change: amount of change in battery, discharging if negative and charging if positive, used to find the battery bin
        #print('battery power before change: ', self._battery_power)
        self._battery_power += change #apply the change to battery power
        self._battery_power = max(self._battery_power, 0) #prevent battery power goes below 0 after applying the change
        #find which bin current power belongs to
        #print('battery power after change:', self._battery_power)
        for each_bin in range(len(self._battery_bin)-1):
            each_bin_min = self._battery_bin[each_bin]
            if self._battery_power >= each_bin_min:
                bin_index = self._battery_bin.index(each_bin_min)
            else: break
        return bin_index

    def get_state(self, change): #combine three pieces of state information
        #change: amount of change in battery
        bin_solar, bin_comed = self.extract_from_table('bin')
        battery_state = self.battery_level(change)
        self._state = [bin_solar, bin_comed, battery_state]

    def get_reward(self, action): #find the reward with current self._state and action
        #action is either 0 (discharging the battery) or 1 (charging the battery)
        p_solar, p_comed = self.extract_from_table('power') #extract exact amount of power generated by solar and consumed
        if action: #if action==1, charging
            reward = min(p_solar, max(0, self._battery_cap - self._battery_power)) #max is to prevent the difference to go below 0 during learning process
        else: #if action==0, discharging
            reward = min(p_comed, self._battery_power) #discharge whatever is consumed or is left in the battery
        #print('power and reward: ', p_solar, p_comed, self._battery_power, reward)
        return reward

    def battery_change(self, action, reward): #find the amount of change in battery
        #action: either 0 (discharging) or 1 (charging)
        if action: #if charging
            return reward
        else: #if discharging
            return reward*(-1)

    def step(self, action): #find the state, reward, whether reached terminal or not after taking giving action at current state
        #action: either 0 (discharging) or 1 (charging)
        #print('before action:', self._state)
        # get reward
        reward = self.get_reward(action) #get reward of current state and action

        #apply changes in state
        battery_change = self.battery_change(action, reward)
        self.get_state(battery_change) #update the state with new data from table and the reward of current step
        #print('after action:', self._state)

        #determine if terminal
        term = True if self._data_step == (self._data.shape[0]-1) else False #if we reach the last row of the dataframe, it's terminal
        self._data_step += 1 #update the row index

        return tuple(self._state), abs(reward), battery_change, term

#testing with first 15 lines of data
# df_join = combine_data()
# split_month()
# plot_df_join(df_join)
# env = solar_power_env()
# combined = env.generate_episode()
# print(combined[8758:8786])
# print(combined[17515:17524])
# print(combined[26278:26284])
# random_actions = np.random.choice([0, 1], 15)
# step = 1
# for eachaction in random_actions:
#     print('step: ', step)
#     print('action: ', eachaction)
#     print(env.step(eachaction))
#     step += 1
