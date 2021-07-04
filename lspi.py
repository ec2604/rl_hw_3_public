import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from mountain_car_with_data_collection import MountainCarWithResetEnv
from data_collector import DataCollector
from data_transformer import DataTransformer
from radial_basis_function_extractor import RadialBasisFunctionExtractor
from linear_policy import LinearPolicy
from game_player import GamePlayer


def compute_lspi_iteration(encoded_states, encoded_next_states, actions, rewards, done_flags, linear_policy, gamma):
    # compute the next w given the data.
    data_subset_idx = np.arange(len(actions))#np.where(actions == linear_policy.get_max_action(encoded_states))
    phi_next_greedy = linear_policy.get_q_features(encoded_next_states[data_subset_idx], linear_policy.get_max_action(encoded_next_states[data_subset_idx]))
    phi_next_greedy[done_flags[data_subset_idx] == 1] = 0
    phi_current = linear_policy.get_q_features(encoded_states[data_subset_idx], actions[data_subset_idx])
    diff = (phi_current - gamma*phi_next_greedy).reshape((len(encoded_states[data_subset_idx]), 1, -1))
    c = np.zeros((phi_current.shape[1],phi_current.shape[1]))
    for i in tqdm(range(len(encoded_states[data_subset_idx]))):
        curr = phi_current[i, :].reshape(-1, 1) @ diff[i, :]
        if c is None:
            c = curr
        else:
            c += curr
    # c = c
    d = np.sum(phi_current*rewards[data_subset_idx].reshape(-1,1),axis=0)
    next_w = np.linalg.inv(c)@d
    return next_w.reshape(-1,1)

def exp_1(env, data_transformer, feature_extractor, evaluation_number_of_games, evaluation_max_steps_per_game, encoded_states,
          encoded_next_states, actions, rewards, done_flags, w_updates, gamma):
    # set a new linear policy
    success_rates_seeds = []

    for i in range(3):
        success_rates = []
        linear_policy = LinearPolicy(feature_extractor.get_number_of_features(), 3, True)
        # but set the weights as random
        linear_policy.set_w(np.random.uniform(size=linear_policy.w.shape))
        # start an object that evaluates the success rate over time
        evaluator = GamePlayer(env, data_transformer, feature_extractor, linear_policy)
        starting_states = np.hstack([np.random.uniform(-1.2, 0.6, (50, 1)), np.zeros((50, 1))])
        starting_states = list(map(tuple, starting_states))
        success_rates.append(
            evaluator.play_games(evaluation_number_of_games, evaluation_max_steps_per_game, starting_states))
        for lspi_iteration in range(w_updates):
            print(f'starting lspi iteration {lspi_iteration}')
            new_w = compute_lspi_iteration(
                encoded_states, encoded_next_states, actions, rewards, done_flags, linear_policy, gamma
            )
            norm_diff = linear_policy.set_w(new_w)
            success_rates.append(evaluator.play_games(evaluation_number_of_games, evaluation_max_steps_per_game, starting_states))
            if norm_diff < 0.00001:
                break
        print('done lspi')
        if len(success_rates) < w_updates:
            success_rates = np.pad(success_rates,pad_width=(0,w_updates+1-len(success_rates)), mode='edge')
        success_rates_seeds.append(success_rates)
    plt.plot(np.arange(w_updates+1), np.mean(success_rates_seeds,axis=0))
    plt.xticks(np.arange(21))
    plt.ylabel('Average success rate')
    plt.xlabel('Iteration number')
    plt.title('Average success rate as a function of LSPI iteration')
    plt.savefig('3_5.png')

def exp_2(env, data_transformer,feature_extractor, evaluation_number_of_games, evaluation_max_steps_per_game, encoded_states,
          encoded_next_states, actions, rewards, done_flags, w_updates, gamma , sample_sizes):
    # set a new linear policy
    success_rates = []
    idx = np.arange(len(encoded_states))
    for sample_size in sample_sizes:
        linear_policy = LinearPolicy(feature_extractor.get_number_of_features(), 3, True)
        # but set the weights as random
        linear_policy.set_w(np.random.uniform(size=linear_policy.w.shape))
        # start an object that evaluates the success rate over time
        evaluator = GamePlayer(env, data_transformer, feature_extractor, linear_policy)
        starting_states = np.hstack([np.random.uniform(-1.2, 0.6, (50, 1)), np.zeros((50, 1))])
        starting_states = list(map(tuple, starting_states))
        sample_idx = idx[:sample_size]
        sampled_encoded_states = encoded_states[sample_idx]
        sampled_encoded_next_states = encoded_next_states[sample_idx]
        sampled_actions = actions[sample_idx]
        sampled_rewards = rewards[sample_idx]
        sampled_done_flags = done_flags[sample_idx]
        for lspi_iteration in range(w_updates):
            print(f'starting lspi iteration {lspi_iteration}')
            new_w = compute_lspi_iteration(
                sampled_encoded_states, sampled_encoded_next_states, sampled_actions, sampled_rewards, sampled_done_flags, linear_policy, gamma
            )
            norm_diff = linear_policy.set_w(new_w)
            if norm_diff < 0.00001:
                break
        print('done lspi')
        success_rates.append(
            evaluator.play_games(evaluation_number_of_games, evaluation_max_steps_per_game, starting_states))
    plt.plot(sample_sizes, success_rates, marker='.')
    plt.ylabel('Success rate after final LSPI iteration')
    plt.xlabel('Sample size')
    plt.title('Success rate as a sample of sample size')
    plt.savefig('3_6.png')

def calc_prelim():
    samples_to_collect = 100000
    # samples_to_collect = 150000
    # samples_to_collect = 10000
    number_of_kernels_per_dim = [12, 10]
    gamma = 0.999
    w_updates = 20
    evaluation_number_of_games = 50
    evaluation_max_steps_per_game = 1000

    np.random.seed(123)
    # np.random.seed(234)

    env = MountainCarWithResetEnv()
    # collect data
    states, actions, rewards, next_states, done_flags = DataCollector(env).collect_data(samples_to_collect)
    # get data success rate
    data_success_rate = np.sum(rewards) / len(rewards)
    print(f'success rate {data_success_rate}')
    # standardize data
    data_transformer = DataTransformer()
    data_transformer.set_using_states(np.concatenate((states, next_states), axis=0))
    states = data_transformer.transform_states(states)
    next_states = data_transformer.transform_states(next_states)
    # process with radial basis functions
    feature_extractor = RadialBasisFunctionExtractor(number_of_kernels_per_dim)
    # encode all states:
    encoded_states = feature_extractor.encode_states_with_radial_basis_functions(states)
    encoded_next_states = feature_extractor.encode_states_with_radial_basis_functions(next_states)
    return env, data_transformer, feature_extractor, evaluation_number_of_games, evaluation_max_steps_per_game, \
           encoded_states, encoded_next_states, encoded_next_states, actions, rewards, done_flags, w_updates, gamma

def run_experiments(run_exp_1=False, run_exp_2=False):
    env, data_transformer, feature_extractor, evaluation_number_of_games, evaluation_max_steps_per_game, encoded_states,\
    encoded_next_states, encoded_next_states, actions, rewards, done_flags, w_updates, gamma = calc_prelim()
    if run_exp_1:
        exp_1(env, data_transformer, feature_extractor, evaluation_number_of_games, evaluation_max_steps_per_game, encoded_states,
              encoded_next_states, actions, rewards, done_flags, w_updates, gamma)
    if run_exp_2:
        exp_2(env, data_transformer,feature_extractor, evaluation_number_of_games, evaluation_max_steps_per_game, encoded_states,
              encoded_next_states, actions, rewards, done_flags, w_updates, gamma,
              np.logspace(start=np.log10(5000), stop=np.log10(100000), num=20, endpoint=True).astype(np.int))

if __name__ == '__main__':
    run_experiments(run_exp_1=True, run_exp_2=True)

