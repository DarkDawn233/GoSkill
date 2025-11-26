import numpy as np

import metaworld

from .wrappers import SeedWrapper

""" constructing envs """

def get_list_of_func_to_make_envs(env_name, task_name_list, config_save_path, seed):

    def get_func_to_make_envs(env_name, task_index, task_name, config_save_path, seed):

        def _make_env():
            if 'metaworld' in env_name:
                mt1 = metaworld.MT1(task_name)
                task_env = mt1.train_classes[task_name]()
                task = mt1.train_tasks[0]
                task_env.set_task(task)
                task_env._freeze_rand_vec = False
            else:
                raise NotImplementedError
            
            task_env = SeedWrapper(task_env)
            task_env.seed(seed)
            return task_env

        return _make_env

    funcs_to_make_envs = []
    for task_idx, task_name in enumerate(task_name_list):
        funcs_to_make_envs.append(get_func_to_make_envs(env_name, task_idx, task_name, config_save_path, seed))

    return funcs_to_make_envs


def get_task_info(env_name, task_name, trajectories, prompt_trajectories):
    if 'metaworld' in env_name:
        max_ep_len = 500
        env_targets = [4500]
        scale = 1000.
    else:
        raise NotImplementedError
    
    if prompt_trajectories is not None:
        if 'rtgs' in prompt_trajectories[0]:
            prompt_rtgs = [prompt_trajectories[i]['rtgs'][0] for i in range(len(prompt_trajectories))]
            env_targets = [np.max(prompt_rtgs)]
    elif trajectories is not None:
        if 'rtgs' in prompt_trajectories[0]:
            rtgs = [trajectories[i]['rtgs'][0] for i in range(len(trajectories))]
            env_targets = [np.max(rtgs)]

    return max_ep_len, env_targets, scale


def get_task_env_and_info(
        env_name,
        task_name_list,
        trajectories_list,
        prompt_trajectories_list,
        config_save_path,
        device,
        seed
    ):
    task_info = {} # store all the attributes for each env

    funcs_to_make_envs = get_list_of_func_to_make_envs(env_name, task_name_list, config_save_path, seed)
    tmp_env = funcs_to_make_envs[0]()
    state_dim = tmp_env.observation_space.shape[0]
    act_dim = tmp_env.action_space.shape[0]
    tmp_env.close()
    
    for i, task_name in enumerate(task_name_list):
        task_info[task_name] = {}
        max_ep_len, env_targets, scale = get_task_info(env_name=env_name, task_name=task_name, trajectories=trajectories_list[i], prompt_trajectories=prompt_trajectories_list[i])
        task_info[task_name]['task_idx'] = i
        task_info[task_name]['max_ep_len'] = max_ep_len
        task_info[task_name]['env_targets'] = env_targets
        task_info[task_name]['scale'] = scale
        task_info[task_name]['state_dim'] = state_dim
        task_info[task_name]['act_dim'] = act_dim
        task_info[task_name]['device'] = device

    return task_info, funcs_to_make_envs