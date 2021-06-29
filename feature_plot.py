import numpy as np
import matplotlib.pyplot as plt

from mountain_car_with_data_collection import MountainCarWithResetEnv
from data_collector import DataCollector
from data_transformer import DataTransformer
from radial_basis_function_extractor import RadialBasisFunctionExtractor


samples_to_collect = 100000
# samples_to_collect = 150000
# samples_to_collect = 10000
number_of_kernels_per_dim = [12, 10]
gamma = 0.99
w_updates = 100
evaluation_number_of_games = 10
evaluation_max_steps_per_game = 1000

np.random.seed(123)
# np.random.seed(234)

feature_extractor = RadialBasisFunctionExtractor(number_of_kernels_per_dim)
position = np.linspace(-1.2, 0.6, 600)
speed = np.linspace(-0.07, 0.07, 600)
# position = np.linspace(-2.5, -1, 600)
# speed = np.linspace(-2, -1, 600)
# encode all states:
xx, yy = np.meshgrid(position, speed)
z = feature_extractor.encode_states_with_radial_basis_functions(np.stack([xx, yy], axis=-1).reshape(-1, 2)).reshape(600,600,-1)
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(position, speed, z[:, :, 0])
plt.show()

###LSPI####
env = MountainCarWithResetEnv()
# collect data
states, actions, rewards, next_states, done_flags = DataCollector(env).collect_data(samples_to_collect)
dt = DataTransformer()
dt.set_using_states(states)