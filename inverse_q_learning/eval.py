import numpy as np

import os
import argparse
import yaml

import numpy as np
import matplotlib.pyplot as plt


def get_arg_parser():

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-conf_file", "--conf_file", help="Configuration file path",default="conf.yaml")
    arg_parser.add_argument("-algo", "--algo", help="Algorithm name",default="iql")

    return arg_parser


def get_conf(conf_path):
    conf_file = open(conf_path)
    conf = yaml.safe_load(conf_file)
    conf_file.close()
    return conf

def action_probs_from_trajectories(trajectories, nS, nA):

    action_probabilities = np.zeros((nS, nA))
    for traj in trajectories:
        for (s, a, r, ns) in traj:
            action_probabilities[s][a] += 1
    action_probabilities[action_probabilities.sum(axis=1) == 0] = 1e-5
    action_probabilities /= action_probabilities.sum(axis=1).reshape(nS, 1)

    return action_probabilities

def load_dataset(data_dir):

    trajectories = np.load(os.path.join(data_dir, "trajectories.npy"))
    transition_probabilities = np.load(os.path.join(data_dir, "transition_probabilities.npy"))
    ground_reward = np.load(os.path.join(data_dir, "ground_reward.npy"))
    feature_matrix = np.load(os.path.join(data_dir, "feature_matrix.npy"))

    (nS, nA, _) = transition_probabilities.shape
    action_probabilities = action_probs_from_trajectories(trajectories, nS, nA)

    return feature_matrix, transition_probabilities, action_probabilities, ground_reward, trajectories


def get_value_map(dis, transition_probabilities, ground_r, thresh, gamma):
    
    nS = transition_probabilities.shape[0]
    policy = dis

    V = np.zeros(nS)

    while True:
        delta = 0
        for s in range(nS): # s = 0
            v = 0
            for a, a_prob in enumerate(policy[s]):
                if a_prob == 0.0:
                    continue
                ns_prob = transition_probabilities[s, a] # 0.2
                next_v = V
                r = ground_r[s]
                v += np.sum(ns_prob * a_prob * (r + gamma * next_v))
            delta = max(delta, np.abs(v - V[s]))
            V[s] = v
        print(delta)
    
        if delta < thresh:
            break
            
    return V

def create_figures(eval_list, optimal_aps, transition_probabilities, ground_reward, thresh = 0.1, gamma = 0.99):
    print("Getting value map for: optimal")
    opt_val_map = get_value_map(optimal_aps, transition_probabilities, ground_reward, thresh, gamma)
    grid = int(np.sqrt(transition_probabilities.shape[0]))
    fig,a =  plt.subplots(1,3)
    for i, eval_path in enumerate(eval_list):
        a[0].imshow(ground_reward.reshape((grid,grid)))
        a[0].set_title('Ground Truth Reward')

        print(f"Getting value map for: {eval_path}")

        try:
            boltz_dis = np.load(os.path.join(eval_path,'boltzman_distribution.npy'))
        except:
            print(os.path.join(eval_path,'boltzman_distribution.npy'), 'is not found. skipping..')
            continue
        val_map = get_value_map(boltz_dis, transition_probabilities, ground_reward, thresh, gamma)
        a[1].imshow(opt_val_map.reshape((grid,grid)))
        a[1].set_title('Optimal Value')

        name = eval_path.split('/')[-2]+" Value"
        a[2].imshow(val_map.reshape((grid,grid)))
        a[2].set_title(name)
        fig.savefig(os.path.join(eval_path,'val_map.png'))

    

        
if __name__=="__main__":

    # Arguiments
    arg_parser = get_arg_parser()
    args = arg_parser.parse_args()

    conf_file = args.conf_file

    # Configuration file
    conf = get_conf(conf_file)

    # directories
    data_dir = conf['dataset']['dataset_path']
    result_folder = conf['train']['result_folder']

    eval_list = conf['eval']['result_folders']
    gamma = conf['env']['gamma']

    # Loading the dataset
    feature_matrix, transition_probabilities, action_probabilities, ground_reward, trajectories = load_dataset(data_dir)
    print(f"Dataset found with trajectories shape of: {trajectories.shape}")

    create_figures(eval_list, action_probabilities, transition_probabilities, ground_reward, thresh = 0.1, gamma = 0.99)
    