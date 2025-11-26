import numpy as np
import pickle
from .others import discount_cumsum

""" data processing """

def load_data_prompt(env_name, task_name_list, data_save_path, data_mode, preprocess_rtg):
    trajectories_list = []
    prompt_trajectories_list = []
    if 'metaworld' in env_name:
        length = 2000 if data_mode == 'expert' else 1000
        for task_name in task_name_list:
            dir_path = data_save_path / "metaworld" / task_name
            cur_trajectories = []
            for i in range(length):
                file_path = dir_path / f"{i}.npz"
                episode = np.load(file_path)
                episode = {k: episode[k] for k in episode.keys()}
                if preprocess_rtg:
                    episode['rtgs'] = discount_cumsum(episode['rewards'], gamma=1.)
                cur_trajectories.append(episode)
            trajectories_list.append(cur_trajectories)

            # load prompt trajectories
            prompt_path = data_save_path / "metaworld" / "prompt" / f"{task_name}-prompt-{data_mode}.npy"
            prompt_trajectories = np.load(prompt_path, allow_pickle=True)
            if preprocess_rtg:
                for episode in prompt_trajectories:
                    episode['rtgs'] = discount_cumsum(episode['rewards'], gamma=1.)
            prompt_trajectories_list.append(prompt_trajectories)
    else:
        for task_name in task_name_list:
            dataset_path = data_save_path / env_name / f'{env_name}-{task_name}-{data_mode}.pkl'
            with open(dataset_path, 'rb') as f:
                trajectories = pickle.load(f)
            if preprocess_rtg:
                for episode in trajectories:
                    episode['rtgs'] = discount_cumsum(episode['rewards'], gamma=1.)
            prompt_dataset_path = data_save_path / env_name / f'{env_name}-{task_name}-prompt-{data_mode}.pkl'
            with open(prompt_dataset_path, 'rb') as f:
                prompt_trajectories = pickle.load(f)
            if preprocess_rtg:
                for episode in prompt_trajectories:
                    episode['rtgs'] = discount_cumsum(episode['rewards'], gamma=1.)
            
            trajectories_list.append(trajectories)
            prompt_trajectories_list.append(prompt_trajectories)
    
    return trajectories_list, prompt_trajectories_list


def process_total_data_mean(trajectories, mode):
    # save all path information into separate lists
    states = []
    for path in trajectories:
        states.append(path['observations'])

    # used for input normalization
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

    return state_mean, state_std


def process_dataset(trajectories, mode, task_name, data_mode, pct_traj):
    # save all path information into separate lists
    states, traj_lens, returns = [], [], []
    for path in trajectories:
        if mode == 'delayed':  # delayed: all rewards moved to end of trajectory
            path['rewards'][-1] = path['rewards'].sum()
            path['rewards'][:-1] = 0.
        states.append(path['observations'])
        traj_lens.append(len(path['observations']))
        returns.append(path['rewards'].sum())
    traj_lens, returns = np.array(traj_lens, dtype=np.int32), np.array(returns, dtype=np.float32)

    # used for input normalization
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

    tot_num_timesteps = sum(traj_lens)

    # only train on top pct_traj trajectories (for %BC experiment)
    used_num_timesteps_max = max(int(pct_traj * tot_num_timesteps), 1)
    sorted_inds = np.argsort(returns)  # lowest to highest
    used_num_trajectories, used_timesteps = 0, 0
    used_trajectories = []
    ind = len(trajectories) - 1
    while ind >= 0 and used_timesteps + traj_lens[sorted_inds[ind]] <= used_num_timesteps_max:
        used_trajectories.append(trajectories[sorted_inds[ind]])
        used_timesteps += traj_lens[sorted_inds[ind]]
        used_num_trajectories += 1
        ind -= 1
    
    sorted_inds = sorted_inds[-used_num_trajectories:]

    # used to reweight sampling so we sample according to timesteps instead of trajectories
    used_p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])
    used_returns = returns[sorted_inds]
    used_sorted_inds = np.argsort(used_returns)

    print('=' * 50)
    print(f'Starting new experiment: {task_name} {data_mode}')
    print(f'{len(traj_lens)} trajectories, {tot_num_timesteps} timesteps found')
    print(f'Using {used_num_trajectories} trajectories, {used_timesteps} timesteps')
    print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    print(f'Average used return: {np.mean(used_returns):.2f}, std: {np.std(used_returns):.2f}')
    print(f'Max used return: {np.max(used_returns):.2f}, min: {np.min(used_returns):.2f}')
    print('=' * 50)

    return used_trajectories, used_num_trajectories, used_sorted_inds, used_p_sample, state_mean, state_std


def process_info(task_name_list, trajectories_list, info, mode, data_mode, pct_traj, variant):
    for i, task_name in enumerate(task_name_list):
        used_trajectories, used_num_trajectories, used_sorted_inds, used_p_sample, state_mean, state_std = process_dataset(
            trajectories=trajectories_list[i], mode=mode, task_name=task_name_list[i], data_mode=data_mode, pct_traj=pct_traj)
        trajectories_list[i] = used_trajectories

        info[task_name]['num_trajectories'] = used_num_trajectories
        info[task_name]['sorted_inds'] = used_sorted_inds
        info[task_name]['p_sample'] = used_p_sample
        if variant['average_state_mean']:
            info[task_name]['state_mean'] = variant['total_state_mean']
            info[task_name]['state_std'] = variant['total_state_std']
        else:
            info[task_name]['state_mean'] = state_mean
            info[task_name]['state_std'] = state_std
    return info
