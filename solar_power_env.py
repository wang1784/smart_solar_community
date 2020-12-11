from get_comed_data import get_comed_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from get_solar_data import get_solar_data
#
def combine_data():
    #get data from comed and solar
    df_comed = get_comed_data().account_hourly() #call function to get comed data
    df_solar = pd.read_csv('solar_data.csv', names = ['Datetime', 'SOLAR_W']).drop(0) #read solar data from csv file
    df_solar['Datetime'] = df_solar['Datetime'].str.replace('-06:00', '') #drop unuseful information so that datetime is consistent with comed data
    df_solar.set_index('Datetime', inplace = True) # set datetime as index
    df_solar[df_solar<0] = 0 #make all the negative values 0 because negative solar generation is meaningless
    df_join = df_comed.join(df_solar, how = 'inner', sort = True) #join two dataframes according to date and time
    df_join['COMED_W'] = df_join['COMED_W'] *2/ 3

    #discretize data
    df_join['solar_bin'] = pd.cut(x=df_join['SOLAR_W'],
                             bins=[-0.75,0,50,100,150,200],
                             labels=[0, 1, 2, 3, 4])
    df_join['SOLAR_W'] = df_join['SOLAR_W'] * 14 #assume each household has 16 panels
    df_join['comed_bin'] = pd.cut(x = df_join['COMED_W'],
                                  bins = np.array([1900, 2800, 3100, 4500, 6500])*2/3,
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

class solar_power_env():
    def __init__(self):
        #get dataframe
        self._data = combine_data()

        #battery
        self._battery_cap = 6000 #max power can be stored in the battery
        self._battery_bin = list(np.linspace(0, self._battery_cap, 6)) #make 5 bins for battery level
        self._battery_power = 0 #records the exact power that the battery has

        #power data
        self._solar_bin = [-0.75,0,50,100,150,200] #bins for solar
        self._comed_bin = np.array([1900, 2800, 3100, 4500, 6500])*2/3 #bins for
        self._data_step = 0 #row index to extract data from dataframe, starting with index 0
        p_init = self._data.iloc[0] #start with first line
        self._state = (p_init['solar_bin'], p_init['comed_bin'], 0) #initiate state

        #state space shape
        self._state_space_shape = [len(self._solar_bin)-1, len(self._comed_bin)-1, len(self._battery_bin)-1]

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
        self.get_state(self.battery_change(action, reward)) #update the state with new data from table and the reward of current step
        #print('after action:', self._state)

        #determine if terminal
        term = True if self._data_step == (self._data.shape[0]-1) else False #if we reach the last row of the dataframe, it's terminal
        self._data_step += 1 #update the row index

        return tuple(self._state), abs(reward), term

#testing with first 15 lines of data
# df_join = combine_data()
# plot_df_join(df_join)
# env = solar_power_env()
# random_actions = np.random.choice([0, 1], 15)
# step = 1
# for eachaction in random_actions:
#     print('step: ', step)
#     print('action: ', eachaction)
#     print(env.step(eachaction))
#     step += 1
