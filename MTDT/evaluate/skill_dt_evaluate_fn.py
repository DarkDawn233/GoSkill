# Code backbone: Decision Transformer https://github.com/kzl/decision-transformer/
# Decision Transformer License: https://github.com/kzl/decision-transformer/blob/master/LICENSE.md

import numpy as np
import torch

from MTDT.envs.vec_env import VecEnv

def skill_dt_evaluate_episode_rtg(
        env,
        task_idxs,
        state_dim,
        act_dim,
        model,
        skill_model,
        max_ep_len=1000,
        scale=1000.,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None,
        mode='normal',
        prompt=None,
        no_state_normalize=False
    ):

    model.eval()
    skill_model.eval()
    model.to(device=device)
    skill_model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device).unsqueeze(1)
    state_std = torch.from_numpy(state_std).to(device=device).unsqueeze(1)

    src_state = env.reset()
    env_num = src_state.shape[0]
    src_state = torch.from_numpy(src_state).to(device=device, dtype=torch.float32).reshape(env_num, 1, -1)
    state = src_state

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    
    cur_state = state
    target_return = torch.from_numpy(target_return).to(device=device, dtype=torch.float32).reshape(env_num, 1, 1)
    ep_return = target_return
    timesteps = torch.zeros((env_num, 1), device=device, dtype=torch.long)

    skill_states = torch.zeros((env_num, 0, cur_state.shape[-1]), device=device, dtype=torch.float32)
    skill_skills = torch.zeros((env_num, 0), device=device, dtype=torch.long)
    skill_skill_encodes = torch.zeros((env_num, 0, skill_model.vq_size), device=device, dtype=torch.float32)
    skill_rtgs = torch.zeros((env_num, 0, 1), device=device, dtype=torch.float32)
    skill_timesteps = torch.zeros((env_num, 0), device=device, dtype=torch.long)

    episode_return = np.zeros(env_num, dtype=np.float32)
    already_done = np.zeros(env_num, dtype=np.bool_)
    episode_length = np.zeros(env_num, dtype=np.int32)

    task_num = env_num
    n_codebook = model.n_codebook
    skill_id = np.zeros((task_num, n_codebook), dtype=np.float32)
    skill_length = model.skill_length

    for t in range(max_ep_len):
        # print('evaluate/t', t)
        # add padding
        if t % skill_length == 0:
            skill_states = torch.cat([skill_states, cur_state], dim=1)
            skill_skills = torch.cat([skill_skills, torch.zeros((env_num, 1), device=device, dtype=torch.long)], dim=1)
            skill_skill_encodes = torch.cat([skill_skill_encodes, torch.zeros((env_num, 1, skill_model.vq_size), device=device, dtype=torch.float32)], dim=1)
            skill_rtgs = torch.cat([skill_rtgs, target_return], dim=1)
            skill_timesteps = torch.cat([skill_timesteps, timesteps // skill_length], dim=1)
            tmp_skill_states = skill_states if no_state_normalize else (skill_states - state_mean) / state_std
            new_skill = model.get_skill(
                tmp_skill_states.to(dtype=torch.float32),
                skill_skills.to(dtype=torch.long),
                skill_skill_encodes.to(dtype=torch.float32),
                skill_rtgs.to(dtype=torch.float32),
                skill_timesteps.to(dtype=torch.long),
                prompt=prompt
            )
            skill_skills[:, -1] = new_skill
            new_skill_encodes = skill_model.vq_codes_to_encodes(new_skill)
            skill_skill_encodes[:, -1] = new_skill_encodes

            skill_id = skill_id + np.eye(n_codebook)[new_skill.cpu().numpy()]

            action_skills = skill_skills[:, -1:]
            action_states = torch.zeros((env_num, 0, cur_state.shape[-1]), device=device, dtype=torch.float32)
            action_actions = torch.zeros((env_num, 0, act_dim), device=device, dtype=torch.float32)
        
        action_states = torch.cat([action_states, cur_state], dim=1)
        action_actions = torch.cat([action_actions, torch.zeros((env_num, 1, act_dim), device=device, dtype=torch.float32)], dim=1)
        tmp_action_states = action_states if no_state_normalize else (action_states - state_mean) / state_std
        new_action = skill_model.get_action(
            tmp_action_states.to(dtype=torch.float32),
            action_actions.to(dtype=torch.float32),
            action_skills.to(dtype=torch.long),
        )
        action_actions[:, -1] = new_action

        action = new_action.detach().cpu().numpy()
        state, reward, done, infos = env.step(action)
        src_state = torch.from_numpy(state).to(device=device, dtype=torch.float32).reshape(env_num, 1, -1)
        cur_state = src_state

        if mode != 'delayed':
            target_return = target_return - torch.from_numpy(reward/scale).to(device=device).unsqueeze(1).unsqueeze(2)

        timesteps = timesteps + 1

        episode_return += reward * (1. - already_done)
        episode_length = episode_length + 1 - (already_done * 1)

        already_done = np.logical_or(already_done, done)

        if already_done.all():
            break

    return_info = {
        'episode_return': episode_return,
        'episode_length': episode_length,
    }

    return episode_return, return_info, skill_id

def skill_dt_eval_vec_episodes(target_rews, info, variant, env_funcs, env_name_list, vec_env_num):

    example_env_name = env_name_list[0]
    for env_name in env_name_list:
        assert info[env_name]['max_ep_len'] == info[example_env_name]['max_ep_len']
        assert info[env_name]['state_dim'] == info[example_env_name]['state_dim']
        assert info[env_name]['act_dim'] == info[example_env_name]['act_dim']
        assert info[env_name]['device'] == info[example_env_name]['device']

    state_means = [info[env_name]['state_mean'] for env_name in env_name_list]
    state_stds = [info[env_name]['state_std'] for env_name in env_name_list]
    scales = [info[env_name]['scale'] for env_name in env_name_list]
    task_num = len(env_name_list)
    
    max_ep_len = info[example_env_name]['max_ep_len']
    state_dim = info[example_env_name]['state_dim']
    act_dim = info[example_env_name]['act_dim']
    device = info[example_env_name]['device']
    num_eval_episodes = variant['num_eval_episodes']
    mode = variant.get('mode', 'normal')

    target_rews = np.array(target_rews, dtype=np.float32)
    scales = np.array(scales, dtype=np.float32)
    state_means = np.array(state_means, dtype=np.float32)
    state_stds = np.array(state_stds, dtype=np.float32)

    def fn(model, skill_model, prompt=None):
        returns = []
        n_codebook = model.n_codebook
        skill_id = np.zeros((task_num, n_codebook), dtype=np.float32)

        for eval_id_begin in range(0, task_num, vec_env_num):
            eval_id_end = min(eval_id_begin + vec_env_num, task_num)
            eval_env_funcs = env_funcs[eval_id_begin: eval_id_end]
            eval_env = VecEnv(
                env_fns=eval_env_funcs,
                context="spawn",
                shared_memory=False,
            )
            eval_returns = []
            eval_task_idxs = [i for i in range(eval_id_begin, eval_id_end)]
            eval_target_rews = target_rews[eval_id_begin: eval_id_end]
            eval_scales = scales[eval_id_begin: eval_id_end]
            eval_prompt = None if prompt is None else [prompt_i[eval_id_begin: eval_id_end] for prompt_i in prompt]

            for i in range(num_eval_episodes):
                with torch.no_grad():
                    ret, infos, skill_id_i = skill_dt_evaluate_episode_rtg(
                        eval_env,
                        eval_task_idxs,
                        state_dim,
                        act_dim,
                        model,
                        skill_model,
                        max_ep_len=max_ep_len,
                        scale=eval_scales,
                        target_return=eval_target_rews / eval_scales,
                        mode=mode,
                        state_mean=state_means[eval_id_begin: eval_id_end],
                        state_std=state_stds[eval_id_begin: eval_id_end],
                        device=device,
                        prompt=eval_prompt,
                        no_state_normalize=variant['no_state_normalize']                
                        )
                eval_returns.append(ret)
            eval_env.close()
            returns.append(np.array(eval_returns, dtype=np.float32))
            
            skill_id[eval_id_begin: eval_id_end] += skill_id_i

        skill_id = np.around(skill_id / skill_id.sum(axis=1, keepdims=True), 2)

        returns = np.concatenate(returns, axis=1)
        return_infos = {}
        for i, env_name in enumerate(env_name_list):
            return_infos.update({
                f'return_mean/{env_name}_target_{target_rews[i]}': np.mean(returns[:, i]),
                f'return_std/{env_name}_target_{target_rews[i]}': np.std(returns[:, i]),
            })
        return_infos.update({
            f'mean_total/return': np.mean(returns[:, :]),
        })
        return return_infos, skill_id
    return fn