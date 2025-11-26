import numpy as np
import torch
from .batch_fns import get_batch

""" prompts """

def flatten_prompt(prompt, batch_size, keys):
    new_prompts = []
    for i, key in enumerate(keys):
        p_data = prompt[i]
        if key in ['observations', 'next_observations', 'actions', 'rewards', 'return_to_gos']:
            p_data = p_data.reshape((batch_size, -1, p_data.shape[-1]))
        elif key in ['terminals', 'timesteps', 'skills']:
            p_data = p_data.reshape((batch_size, -1))
        elif key in ['task_index']:
            p_data = p_data.reshape((batch_size))
        else:
            raise NotImplementedError
        new_prompts.append(p_data)

    p_mask = prompt[-1].reshape((batch_size, -1))
    new_prompts.append(p_mask)
    return new_prompts


def get_prompt(prompt_trajectories, info, stochastic_prompt, no_state_normalize, keys=None, interval_length=None):
    num_trajectories, p_sample, sorted_inds = info['num_trajectories'], info['p_sample'], info['sorted_inds']
    max_ep_len, state_mean, state_std, scale = info['max_ep_len'], info['state_mean'], info['state_std'], info['scale']
    state_dim, act_dim, device = prompt_trajectories[0]['observations'].shape[-1], prompt_trajectories[0]['actions'].shape[-1], info['device']
    task_idx = info['task_idx']

    state_mean = state_mean.reshape(1, -1)
    state_std = state_std.reshape(1, -1)

    interval_length = interval_length if interval_length is not None else 1

    def fn(sample_size, num_episode, max_len, skip_len=1):
        # random sample prompts with fixed length (prompt-length) in num episodes (prompt-episode)
        batch_inds = np.random.choice(
            np.arange(len(prompt_trajectories)),
            size=int(num_episode*sample_size),
            replace=True,
            # p=p_sample,  # reweights so we sample according to timesteps
        )

        s = np.zeros((num_episode*sample_size, max_len, state_dim), dtype=np.float32) if "observations" in keys else None
        nxt_s = np.zeros((num_episode*sample_size, max_len, state_dim), dtype=np.float32) if "next_observations" in keys else None
        a = np.zeros((num_episode*sample_size, max_len, act_dim), dtype=np.float32) if "actions" in keys else None
        r = np.zeros((num_episode*sample_size, max_len, 1), dtype=np.float32) if "rewards" in keys else None
        d = np.ones((num_episode*sample_size, max_len), dtype=np.float32) * 2 if "terminals" in keys else None
        rtg = np.zeros((num_episode*sample_size, max_len, 1), dtype=np.float32) if "return_to_gos" in keys else None # Why +1
        timesteps = np.zeros((num_episode*sample_size, max_len), dtype=np.int32) if "timesteps" in keys else None
        mask = np.zeros((num_episode*sample_size, max_len), dtype=np.float32)
        skill = np.zeros((num_episode*sample_size, max_len), dtype=np.int32) if "skills" in keys else None

        for i in range(int(num_episode*sample_size)):
            if stochastic_prompt:
                traj = prompt_trajectories[int(batch_inds[i])] # random select traj
            else:
                traj = prompt_trajectories[int(sorted_inds[-i])] # select the best traj with highest rewards
                # traj = prompt_trajectories[i]

            si_end = traj['rewards'].shape[0]
            if si_end < max_len * interval_length:
                si_begin = si_end - si_end // interval_length * interval_length
            else:
                si_begin = si_end - max_len * interval_length

            si_idx = np.arange(si_begin, si_end, interval_length)
            
            # get sequences from dataset
            if "observations" in keys:
                s_i = traj['observations'][si_idx].reshape(-1, state_dim)
                s_i = (s_i - state_mean) / state_std if not no_state_normalize else s_i
                s[i, -s_i.shape[0]:] = s_i
                    
            if "next_observations" in keys:
                nxt_si_idx = si_idx + interval_length - 1
                if nxt_si_idx[-1] >= traj['rewards'].shape[0]:
                    nxt_si_idx[-1] = traj['rewards'].shape[0] - 1
                nxt_s_i = traj['next_observations'][nxt_si_idx].reshape(-1, state_dim)
                nxt_s_i = (nxt_s_i - state_mean) / state_std if not no_state_normalize else nxt_s_i
                nxt_s[i, -nxt_s_i.shape[0]:] = nxt_s_i

            if "actions" in keys:
                a_i = traj['actions'][si_idx].reshape(-1, act_dim)
                a[i, -a_i.shape[0]:] = a_i

            if "rewards" in keys:
                r_i = traj['rewards'][si_idx].reshape(-1, 1)
                r[i, -r_i.shape[0]:] = r_i

            if "terminals" in keys:
                if 'terminals' in traj:
                    d_i = traj['terminals'][si_idx].reshape(-1)
                else:
                    d_i = traj['dones'][si_idx].reshape(-1)
                d[i, -d_i.shape[0]:] = d_i

            if "timesteps" in keys:
                timesteps_i = si_idx // interval_length
                timesteps[i, -timesteps_i.shape[0]:] = timesteps_i

            if "return_to_gos" in keys:
                rtg_i = traj['rtgs'][si_idx].reshape(-1, 1)
                rtg[i, -rtg_i.shape[0]:] = rtg_i / scale

            if "skills" in keys:
                skill_i = traj['skills'][si_idx].reshape(-1)
                skill[i, -skill_i.shape[0]:] = skill_i
                assert -1 not in skill_i

            mask[i, -s_i.shape[0]:] = 1

        s = torch.from_numpy(s).to(dtype=torch.float32, device=device) if "observations" in keys else None
        nxt_s = torch.from_numpy(nxt_s).to(dtype=torch.float32, device=device) if "next_observations" in keys else None
        a = torch.from_numpy(a).to(dtype=torch.float32, device=device) if "actions" in keys else None
        r = torch.from_numpy(r).to(dtype=torch.float32, device=device) if "rewards" in keys else None
        d = torch.from_numpy(d).to(dtype=torch.long, device=device) if "terminals" in keys else None
        rtg = torch.from_numpy(rtg).to(dtype=torch.float32, device=device) if "return_to_gos" in keys else None
        timesteps = torch.from_numpy(timesteps).to(dtype=torch.long, device=device) if "timesteps" in keys else None
        mask = torch.from_numpy(mask).to(device=device)
        task_id = torch.ones((num_episode*sample_size)).to(dtype=torch.long, device=device) * task_idx if "task_index" in keys else None
        skill = torch.from_numpy(skill).to(dtype=torch.long, device=device) if "skills" in keys else None

        prompt_data = []
        for key in keys:
            if key == "observations":
                prompt_data.append(s)
            elif key == "next_observations":
                prompt_data.append(nxt_s)
            elif key == "actions":
                prompt_data.append(a)
            elif key == "rewards":
                prompt_data.append(r)
            elif key == "terminals":
                prompt_data.append(d)
            elif key == "return_to_gos":
                prompt_data.append(rtg)
            elif key == "timesteps":
                prompt_data.append(timesteps)
            elif key == "task_index":
                prompt_data.append(task_id)
            elif key == "skills":
                prompt_data.append(skill)
            else:
                raise NotImplementedError
        prompt_data.append(mask)
        
        return prompt_data

    return fn

def get_all_prompt(prompt_trajectories_list, info, task_name_list, stochastic_prompt, no_state_normalize, keys=None, interval_length=None):
    
    def fn(sample_size, num_episodes, max_len):
        prompt_lists = []
        for _ in range(len(keys)+1):
            prompt_lists.append([])

        for task_id, task_name in enumerate(task_name_list):
            get_prompt_fn = get_prompt(prompt_trajectories_list[task_id], info[task_name], stochastic_prompt, no_state_normalize, keys=keys, interval_length=interval_length)
            prompt = flatten_prompt(get_prompt_fn(sample_size, num_episodes, max_len), sample_size, keys=keys)
            
            for i, p_data in enumerate(prompt):
                prompt_lists[i].append(p_data)
            
        for i in range(len(prompt_lists)):
            prompt_lists[i] = torch.cat(prompt_lists[i], dim=0)

        return prompt_lists
    
    return fn

def get_prompt_batch(trajectories_list, prompt_trajectories_list, info, train_task_name_list,
                     stochastic_prompt, no_state_normalize, keys=None, interval_length=None):

    def fn(batch_size, max_len, prompt_episode, prompt_length, no_prompt=False,
           l_blank=0, r_blank=0, skip_len=1, trajectory_ids=None):
        data_lists = []
        prompt_lists = []
        for _ in range(len(keys)+1):
            data_lists.append([])
            prompt_lists.append([])

        for task_id, task_name in enumerate(train_task_name_list):
            if not no_prompt:
                if prompt_trajectories_list:
                    get_prompt_fn = get_prompt(prompt_trajectories_list[task_id], info[task_name], stochastic_prompt, no_state_normalize, keys=keys, interval_length=interval_length)
                else:
                    get_prompt_fn = get_prompt(trajectories_list[task_id], info[task_name], stochastic_prompt, no_state_normalize, keys=keys, interval_length=interval_length)
                prompt = flatten_prompt(get_prompt_fn(sample_size=batch_size, num_episode=prompt_episode, max_len=prompt_length), batch_size, keys=keys)
                
                for i, p_data in enumerate(prompt):
                    prompt_lists[i].append(p_data)


            get_batch_fn = get_batch(trajectories_list[task_id], info[task_name], no_state_normalize, keys=keys, interval_length=interval_length)
            batch = get_batch_fn(batch_size=batch_size, max_len=max_len, l_blank=l_blank, r_blank=r_blank, skip_len=skip_len, trajectory_ids=trajectory_ids)
            
            for i, d_data in enumerate(batch):
                data_lists[i].append(d_data)

        if not no_prompt:
            for i in range(len(prompt_lists)):
                prompt_lists[i] = torch.cat(prompt_lists[i], dim=0)
            prompt = prompt_lists
        else:
            prompt = None

        for i in range(len(data_lists)):
            data_lists[i] = torch.cat(data_lists[i], dim=0)
        batch = data_lists

        return prompt, batch
    return fn


def get_batch_for_skill(trajectories_list, prompt_trajectories_list, info, train_task_name_list,
                     stochastic_prompt, no_state_normalize, keys=None, interval_length=None, skill_index_in_data=None):

    def fn(batch_size, max_len, prompt_episode, prompt_length, no_prompt=False, l_blank=0, r_blank=0, skip_len=1, trajectory_ids=None):
        assert no_prompt == True
        data_lists = []
        for _ in range(len(keys)+1):
            data_lists.append([])

        valid_skill_list = [skill_index_in_data_i is not None for skill_index_in_data_i in skill_index_in_data]
        valid_skill_num = np.sum(valid_skill_list)
        batch_size_for_single_skill = batch_size * len(train_task_name_list) // valid_skill_num

        for skill_id in range(len(skill_index_in_data)):
            all_sample_ids = skill_index_in_data[skill_id]
            if all_sample_ids is None:
                continue
            skill_sample_ids = all_sample_ids[np.random.choice(len(all_sample_ids), size=batch_size_for_single_skill, replace=True)]
            # sample_ids = sorted(sample_ids, key=lambda x: x[0])
            for task_id, task_name in enumerate(train_task_name_list):
                sample_ids = skill_sample_ids[skill_sample_ids[:, 0] == task_id][:, 1:]
                if len(sample_ids) == 0:
                    continue
                get_batch_fn = get_batch(trajectories_list[task_id], info[task_name], no_state_normalize, keys=keys, interval_length=interval_length)
                batch = get_batch_fn(batch_size=len(sample_ids), max_len=max_len, l_blank=l_blank, r_blank=r_blank, skip_len=skip_len, sample_ids=sample_ids)
                
                for i, d_data in enumerate(batch):
                    data_lists[i].append(d_data)

        prompt = None

        for i in range(len(data_lists)):
            data_lists[i] = torch.cat(data_lists[i], dim=0)
        batch = data_lists

        return prompt, batch
    
    return fn