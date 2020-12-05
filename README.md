# Smart Solar Community

## Table of contents
* [General info](#general-info)
* [Data Sources](#technologies)
* [Files](#files)
* [Sources](#setup)

## General
Reinforcement learning algorithm that optimizes the distribution of solar power within a community that shares solar so that the total amount of electricity drawn from the grid is minimized. 

## Data Sources
* Install the PVlib library to generate solar [data](https://pvlib-python.readthedocs.io/en/latest/introtutorial.html)
* Existing electricity usage data taken from the COMED [region](https://www.kaggle.com/robikscube/hourly-energy-consumption)

## Files
* [Comed data](get_comed_data.py): Gets usage data
* [Solar data](get_solar_data.py): Generates solar data from 2011 to 2018 from PVlib
* [RL agent](Q_learning.py): Class containing Q_learning agent
* [Environment](solor_power_env.py): Class that sets up solar environment and functions to get state and reward


## Sources
This approach is inspired by a [paper](https://www.researchgate.net/publication/283653856_Multi_agent_reinforcement_learning_based_distributed_optimization_of_solar_microgrid) by R. Leo, R. S. Milton and A. Kaviya, "Multi agent reinforcement learning based distributed optimization of solar microgrid"
