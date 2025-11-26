# Code backbone: Decision Transformer https://github.com/kzl/decision-transformer/
# Decision Transformer License: https://github.com/kzl/decision-transformer/blob/master/LICENSE.md

import numpy as np
import torch
import torch.nn.functional as F
import time
from MTDT.utils.prompt_fns import get_prompt_batch, get_batch_for_skill
from MTDT.utils.batch_fns import get_batch

class SkillQuantizedTrainer:

    def __init__(
            self,
            model,
            config
        ):
        self.model = model
        self.config = config
        self._get_optimizer()
        self.batch_size = self.config['batch_size']
        self.max_norm = self.config['max_norm']
        self.vq_coef = self.config['vq_coef']
        self.skill_length = model.skill_length
        self.augment = config['augment']
        self.diagnostics = dict()
   
        self.start_time = time.time()


    def pure_train_iteration_mix(self, only_train_decode=False):

        logs = dict()
        train_start = time.time()
        self.model.train()
        train_loss, vq_loss, pre_loss  = self.train_step_mix(only_train_decode)
        if self.scheduler is not None:
            self.scheduler.step()

        logs[f'skill_quantized_training/time'] = time.time() - train_start
        logs[f'skill_quantized_training/train_loss_mean'] = train_loss
        logs[f'skill_quantized_training/vq_loss_mean'] = vq_loss
        logs[f'skill_quantized_training/loss_mean'] = pre_loss

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        return logs
    

    def train_step_mix(self, only_train_decode=False):
        if self.augment:
            skip_len = 1
        else:
            skip_len = self.skill_length

        _, batch = self.prompt_batch_fn(
            batch_size=self.batch_size,
            max_len=self.skill_length,
            prompt_episode=None,
            prompt_length=None,
            no_prompt=True,
            l_blank=0, r_blank=self.skill_length - 1, skip_len=skip_len
        )

        states, nxt_states, actions, attention_mask = batch
        action_target = torch.clone(actions)

        freeze_codebook = True if only_train_decode else False
        action_preds, skill_vq_codes, vq_loss, all_encodes = self.model(states, nxt_states, actions, attention_mask, freeze_codebook=freeze_codebook)

        pre_loss = torch.mean((action_preds-action_target)**2, dim=-1)

        pre_loss = pre_loss.reshape(-1)[attention_mask.reshape(-1) > 0].mean()

        loss = pre_loss
        if not only_train_decode:
            loss = loss + self.vq_coef * vq_loss

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics['skill_quantized_training/action_error'] = pre_loss.detach().cpu().item()
            self.diagnostics['skill_quantized_training/grad_norm'] = grad_norm.detach().cpu().item()
            self.diagnostics['skill_quantized_training/learning_rate'] = self.optimizer.state_dict()['param_groups'][0]['lr']
            
        return loss.detach().cpu().item(), vq_loss.detach().cpu().item(), pre_loss.detach().cpu().item()
    

    def _get_optimizer(self):
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay'],
        )
        warmup_steps = self.config['warmup_steps']
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: min(1.0, (step + 1) / warmup_steps)
        )

    
    def get_prompt_batch_fn(
            self,
            trajectories_list,
            prompt_trajectories_list,
            train_info,
            train_task_name_list,
            no_state_normalize,
            resample = False,
            skill_index_in_data = None
            ):
        self.data_keys = ['observations', 'next_observations', 'actions']
        if resample:
            assert skill_index_in_data is not None
            self.prompt_batch_fn = get_batch_for_skill(
                trajectories_list, None, train_info, train_task_name_list,
                None, no_state_normalize, keys=self.data_keys,
                skill_index_in_data=skill_index_in_data
            )
        else:
            self.prompt_batch_fn = get_prompt_batch(
                trajectories_list, None, train_info, train_task_name_list,
                None, no_state_normalize, keys=self.data_keys
            )


    def get_model_path(self, env_name, postfix, folder):
        model_name = 'skill_quantized_model_' + env_name + postfix
        model_path = folder / model_name
        return model_path


    def save_model(self, env_name, postfix, folder):
        model_path = self.get_model_path(env_name, postfix, folder)
        torch.save(self.model.state_dict(), model_path)  # model save
        print('model saved to ', model_path)


    def load_model(self, env_name, postfix, folder):
        model_path = self.get_model_path(env_name, postfix, folder)
        self.model.load_state_dict(torch.load(model_path))
        print('model initialized from: ', model_path)
        
    
    def preprocess_skills(self, trajectories_list, prompt_trajectories_list, task_name_list, task_info_dict, no_state_normalize, preprocess_traj_num=50):
        self.model.eval()
        skill_length = self.model.skill_length
        skill_n = self.model.n_codebook
        batch_keys = ['observations', 'next_observations']
        skill_index_in_data = {i : [] for i in range(skill_n)}
        two_traj_lists = [trajectories_list, prompt_trajectories_list]

        if self.augment:
            skip_len = 1
        else:
            skip_len = skill_length

        for traj_idx, traj_list in enumerate(two_traj_lists):
            if traj_list is None:
                continue
            for task_idx, task_trajectories in enumerate(traj_list):
                trajectory_num = len(task_trajectories)
                task_name = task_name_list[task_idx]
                task_info = task_info_dict[task_name]
                for begin_id in range(0, trajectory_num, preprocess_traj_num):
                    end_id = min(begin_id + preprocess_traj_num, trajectory_num)
                    trajectory_ids = [i for i in range(begin_id, end_id)]
                    sample_ids = []
                    skill_nums = []
                    traj_lengths = []
                    for trajectory_i in trajectory_ids:
                        traj_length = task_trajectories[trajectory_i]['rewards'].shape[0]
                        ts = np.arange(0, traj_length, skip_len)
                        skill_nums.append(len(ts))
                        traj_lengths.append(traj_length)
                        sample_ids.append(np.stack([np.ones_like(ts) * trajectory_i, ts], axis=1))
                    sample_ids = np.concatenate(sample_ids, axis=0)

                    batch_size = sample_ids.shape[0]

                    get_batch_fn = get_batch(task_trajectories, task_info, no_state_normalize, 
                                             keys=batch_keys, interval_length=skill_length)
                    batch = get_batch_fn(
                        batch_size=batch_size,
                        max_len=1,
                        l_blank=0, r_blank=self.skill_length - 1, skip_len=1,
                        sample_ids=sample_ids
                    )
                    
                    states, nxt_states, attention_mask = batch
                    bs, seq_len, _ = states.shape
                    assert seq_len == 1
                    skill_states = states.reshape(bs, 1, -1)
                    skill_nxt_states = nxt_states.reshape(bs, 1, -1)
                    skills, _ = self.model.get_skill_vq_codes_encodes(skill_states, skill_nxt_states)
                    
                    skills = skills.reshape(bs)
                    skills = skills.detach().cpu().numpy()
                    skill_list = np.split(skills, np.cumsum(skill_nums)[:-1])

                    for i, skill_i in enumerate(skill_list):
                        traj_length = traj_lengths[i]
                        skill_data = -1 * np.ones(traj_length)
                        skill_data[:: skip_len] = skill_i
                        task_trajectories[i + begin_id]['skills'] = skill_data

                    if traj_idx != 0:
                        continue

                    # process skill_index for trajectories_list
                    for i in range(skill_n):
                        for j, skill_i in enumerate(skill_list):
                            skill_index_in_data_i = np.argwhere(task_trajectories[j + begin_id]['skills'] == i)
                            if len(skill_index_in_data_i) == 0:
                                continue
                            traj_ids = np.ones_like(skill_index_in_data_i) * (j + begin_id)
                            task_ids = np.ones_like(skill_index_in_data_i) * task_idx
                            skill_index_in_data_i = np.concatenate([task_ids, traj_ids, skill_index_in_data_i], axis=1)
                            skill_index_in_data[i].append(skill_index_in_data_i)

        for i in range(skill_n):
            if len(skill_index_in_data[i]) == 0:
                skill_index_in_data[i] = None
            else:
                skill_index_in_data[i] = np.concatenate(skill_index_in_data[i], axis=0)

        tmp = [len(skill_index_in_data[i]) if skill_index_in_data[i] is not None else 0 for i in range(skill_n)]
        print("skill proportion:", (np.array(tmp) / np.sum(tmp)).tolist())

        return trajectories_list, prompt_trajectories_list, skill_index_in_data
