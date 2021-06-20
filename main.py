import numpy as np
from matplotlib import pyplot as plt
from gym import spaces

from mountain_car_with_data_collection import MountainCarWithResetEnv
from data_collector import DataCollector
from data_transformer import DataTransformer
from radial_basis_function_extractor import RadialBasisFunctionExtractor
from linear_policy import LinearPolicy
from game_player import GamePlayer

number_of_kernels_per_dim = [12, 10]
feature_extractor = RadialBasisFunctionExtractor(number_of_kernels_per_dim)
min_position = -1.2
max_position = 0.6
max_speed = 0.07
resolution = 100
x_vec = np.linspace(min_position, max_position, resolution)
v_vec = np.linspace(-max_speed, max_speed, resolution)

xx, vv = np.meshgrid(x_vec, v_vec)
states = np.stack((xx, vv), axis=2)
states_coords = states.reshape(resolution**2, 2)
feats = feature_extractor.encode_states_with_radial_basis_functions(states_coords)
feats = feats.reshape(resolution, resolution, -1)
# feats_sorted = np.sort(feats, axis=2) 
feats_sorted = feats
feat0 = feats_sorted[:,:,-1]
fig = plt.figure()
ax = plt.axes(projection='3d')
h = ax.contour3D(xx, vv, feat0)
# plt.show()

feat1 = feats_sorted[:,:,-2]
fig = plt.figure()
ax = plt.axes(projection='3d')
h = ax.contour3D(xx, vv, feat1)
plt.show()

print('ya')