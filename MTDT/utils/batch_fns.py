import numpy as np
import torch
import random
from .others import discount_cumsum

""" batches """

def get_batch(trajectories, info, no_state_normalize, keys=None, interval_length=None):
    num_trajectories, p_sample, sorted_inds = info['num_trajectories'], info['p_sample'], info['sorted_inds']
    max_ep_len, state_mean, state_std, scale = info['max_ep_len'], info['state_mean'], info['state_std'], info['scale']
    state_dim, act_dim, device = trajectories[0]['observations'].shape[-1], trajectories[0]['actions'].shape[-1], info['device']
    task_idx = info['task_idx']

    state_mean = state_mean.reshape(1, -1)
    state_std = state_std.reshape(1, -1)

    interval_length = interval_length if interval_length is not None else 1

    def fn(batch_size, max_len, l_blank=0, r_blank=0, skip_len=1, trajectory_ids=None, sample_ids=None):
        si_begins = None
        if sample_ids is not None:
            batch_inds = sample_ids[:, 0]
            assert len(batch_inds) == batch_size
            si_begins = sample_ids[:, 1]
        elif trajectory_ids is not None:
            batch_inds = trajectory_ids
            assert len(batch_inds) == batch_size
        else:
            batch_inds = np.random.choice(
                np.arange(num_trajectories),
                size=batch_size,
                replace=True,
                p=p_sample,  # reweights so we sample according to timesteps
            )

        # initialize arrays without concat for acceleration
        s = np.zeros((batch_size, max_len, state_dim), dtype=np.float32) if "observations" in keys else None
        nxt_s = np.zeros((batch_size, max_len, state_dim), dtype=np.float32) if "next_observations" in keys else None
        a = np.zeros((batch_size, max_len, act_dim), dtype=np.float32) if "actions" in keys else None
        r = np.zeros((batch_size, max_len, 1), dtype=np.float32) if "rewards" in keys else None
        d = np.ones((batch_size, max_len), dtype=np.float32) * 2 if "terminals" in keys else None
        rtg = np.zeros((batch_size, max_len, 1), dtype=np.float32) if "return_to_gos" in keys else None # Why +1
        timesteps = np.zeros((batch_size, max_len), dtype=np.int32) if "timesteps" in keys else None
        mask = np.zeros((batch_size, max_len), dtype=np.float32)
        skill = np.zeros((batch_size, max_len), dtype=np.int32) if "skills" in keys else None

        for i in range(batch_size):
            traj = trajectories[int(batch_inds[i])]
            if si_begins is not None:
                si_begin = si_begins[i]
            else:
                min_si_begin = -l_blank * interval_length
                max_si_begin = traj['rewards'].shape[0] + (r_blank + 1) * interval_length - 1 - max_len * interval_length
                si_begin = random.randint(min_si_begin, max_si_begin)
                si_begin = si_begin // skip_len * skip_len 
                
            si_end = si_begin + max_len * interval_length

            if si_begin < 0:
                tmp = si_begin // interval_length
                tar_begin = -tmp
                si_begin = si_begin + (-tmp) * interval_length
            else:
                tar_begin = 0
            if si_end > traj['rewards'].shape[0]:
                tar_end = max_len - (si_end - traj['rewards'].shape[0]) // interval_length
                si_end = traj['rewards'].shape[0]
            else:
                tar_end = max_len
            
            si_idx = np.arange(si_begin, si_end, interval_length)

            # get sequences from dataset
            if "observations" in keys:
                s_i = traj['observations'][si_idx].reshape(-1, state_dim)
                s_i = (s_i - state_mean) / state_std if not no_state_normalize else s_i
                s[i, tar_begin: tar_end] = s_i

            if "next_observations" in keys:
                nxt_si_idx = si_idx + interval_length - 1
                if nxt_si_idx[-1] >= traj['rewards'].shape[0]:
                    nxt_si_idx[-1] = traj['rewards'].shape[0] - 1

                nxt_s_i = traj['next_observations'][nxt_si_idx].reshape(-1, state_dim)
                nxt_s[i, tar_begin: tar_end] = (nxt_s_i - state_mean) / state_std if not no_state_normalize else nxt_s_i
                if tar_end != max_len:
                    nxt_s[i, max_len-1] = nxt_s_i[-1]

            if "actions" in keys: 
                a_i = traj['actions'][si_idx].reshape(-1, act_dim)
                a[i, tar_begin: tar_end] = a_i

            if "rewards" in keys:
                r_i = traj['rewards'][si_idx].reshape(-1, 1)
                r[i, tar_begin: tar_end] = r_i

            if "terminals" in keys:
                if 'terminals' in traj:
                    d_i = traj['terminals'][si_idx].reshape(-1)
                else:
                    d_i = traj['dones'][si_idx].reshape(-1)
                d[i, tar_begin: tar_end] = d_i

            if "timesteps" in keys:
                timesteps_i = si_idx // interval_length
                timesteps[i, tar_begin: tar_end] = timesteps_i

            if "return_to_gos" in keys:
                rtg_i = traj['rtgs'][si_idx].reshape(-1, 1)
                rtg[i, tar_begin: tar_end] = rtg_i / scale
            
            if "skills" in keys:
                skill_i = traj['skills'][si_idx].reshape(-1)
                skill[i, tar_begin: tar_end] = skill_i
                assert -1 not in skill_i

            mask[i, tar_begin:tar_end] = 1

        s = torch.from_numpy(s).to(dtype=torch.float32, device=device) if "observations" in keys else None
        nxt_s = torch.from_numpy(nxt_s).to(dtype=torch.float32, device=device) if "next_observations" in keys else None
        a = torch.from_numpy(a).to(dtype=torch.float32, device=device) if "actions" in keys else None
        r = torch.from_numpy(r).to(dtype=torch.float32, device=device) if "rewards" in keys else None
        d = torch.from_numpy(d).to(dtype=torch.long, device=device) if "terminals" in keys else None
        rtg = torch.from_numpy(rtg).to(dtype=torch.float32, device=device) if "return_to_gos" in keys else None
        timesteps = torch.from_numpy(timesteps).to(dtype=torch.long, device=device) if "timesteps" in keys else None
        mask = torch.from_numpy(mask).to(device=device) # TODO: why mask only has several zeros
        task_id = torch.ones((batch_size)).to(dtype=torch.long, device=device) * task_idx if "task_index" in keys else None
        skill = torch.from_numpy(skill).to(dtype=torch.long, device=device) if "skills" in keys else None

        batch_data = []
        for key in keys:
            if key == "observations":
                batch_data.append(s)
            elif key == "next_observations":
                batch_data.append(nxt_s)
            elif key == "actions":
                batch_data.append(a)
            elif key == "rewards":
                batch_data.append(r)
            elif key == "terminals":
                batch_data.append(d)
            elif key == "return_to_gos":
                batch_data.append(rtg)
            elif key == "timesteps":
                batch_data.append(timesteps)
            elif key == "task_index":
                batch_data.append(task_id)
            elif key == "skills":
                batch_data.append(skill)
            else:
                raise NotImplementedError
        batch_data.append(mask)
        
        return batch_data

    return fn

