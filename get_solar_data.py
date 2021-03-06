#testing local changes
#import libraries needed
from pvlib.pvsystem import PVSystem
from pvlib.location import Location
from pvlib.modelchain import ModelChain
import pvlib
import pandas as pd
# import matplotlib.pyplot as plt

class solar_data():
	def get_solar_data(self):
		### Set up parameters for PV model ###
		#copied from examples for the following lines
		sandia_modules = pvlib.pvsystem.retrieve_sam('SandiaMod')
		module = sandia_modules['Canadian_Solar_CS5P_220M___2009_']
		sapm_inverters = pvlib.pvsystem.retrieve_sam('cecinverter')
		temperature_model_parameters = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']
		inverter = sapm_inverters['ABB__MICRO_0_25_I_OUTD_US_208__208V_']
		system = PVSystem(module_parameters=module, inverter_parameters=inverter,
						  temperature_model_parameters=temperature_model_parameters)

		#set up location and time ###
		coordinates = [(41.881832, -87.623177, 'Chicago', 594, 'Etc/GMT+6')] #set up chicago as location
		naive_times = pd.date_range(start='2011', end='2018', freq='1h') #set up time frame and frequency as 1 hour

		### Get hourly solar power generation ###
		for latitude, longitude, name, altitude, timezone in coordinates:
				times = naive_times.tz_localize(timezone)
				location = Location(latitude, longitude, name=name, altitude=altitude,tz=timezone)
				weather = location.get_clearsky(times)
				mc = ModelChain(system, location, orientation_strategy='south_at_latitude_tilt')
				mc.run_model(weather)
		self.df = pd.DataFrame(mc.ac)


	def output_to_csv(self):
		print('Doing it')
		self.df.to_csv('~/Documents/GitHub/smart_solar_community/solar_data.csv')

	def bins(self):
		df = get_solar_data()
		df.columns = ['solar_power']
		# print(df.describe())
		#plt.bar(df['2013'].index, df['2013'].solar_power)
		#plt.show()
		df['binned'] = pd.cut(x=df['solar_power'], bins=[-0.75,0,50,100,150,200,250])
		df['solar_bin'] = pd.cut(x = df['solar_power'],
								bins = [-0.75,0,50,100,150,200,250],
								labels = [0, 1, 2,3,4,5])
		print(df['solar_bin'].value_counts())

test= solar_data()
test.get_solar_data()
test.output_to_csv()