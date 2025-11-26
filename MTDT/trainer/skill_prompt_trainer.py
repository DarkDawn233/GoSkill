# Code backbone: Decision Transformer https://github.com/kzl/decision-transformer/
# Decision Transformer License: https://github.com/kzl/decision-transformer/blob/master/LICENSE.md

import numpy as np
import torch
import torch.nn.functional as F
import time

from MTDT.utils.prompt_fns import get_prompt_batch, get_all_prompt
from MTDT.evaluate.skill_dt_evaluate_fn import skill_dt_eval_vec_episodes


class SkillPromptSequenceTrainer:

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
        self.max_length = self.config['max_length']
        self.augment = config['augment']
        self.skill_length = self.model.skill_length
        self.embed_skill_encode = self.model.embed_skill_encode
        self.prompt_episode = self.config['prompt_episode']
        self.prompt_length = self.config['prompt_length']
        self.diagnostics = dict()

        self.start_time = time.time()


    def pure_train_iteration_mix(self, skill_model):

        logs = dict()
        train_start = time.time()

        self.model.train()
        skill_model.eval()
        train_loss = self.train_step_mix(skill_model)
        if self.scheduler is not None:
            self.scheduler.step()

        logs['skill_dt_training/time'] = time.time() - train_start
        logs['skill_dt_training/train_loss_mean'] = train_loss

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        return logs


    def train_step_mix(self, skill_model):
        if self.augment:
            skip_len = 1
        else:
            skip_len = self.skill_length

        no_prompt=self.config['no_prompt']
        prompt, batch = self.prompt_batch_fn(
            batch_size=self.batch_size,
            max_len=self.max_length,
            prompt_episode=self.prompt_episode,
            prompt_length=self.prompt_length,
            no_prompt=no_prompt,
            l_blank=self.max_length-1, r_blank=0, skip_len=skip_len
        )

        if self.config['preprocess_skill']:
            if self.model.no_rtg:
                states, skills, timesteps, attention_mask = batch
                rtg = None
            else:
                states, skills, rtg, timesteps, attention_mask = batch
            
            if self.embed_skill_encode:
                skill_encodes = skill_model.vq_codes_to_encodes(skills)
                if self.model.no_rtg:
                    prompt_states, prompt_skills, prompt_timesteps, prompt_attention_mask = prompt
                else:
                    prompt_states, prompt_skills, prompt_returns_to_go, prompt_timesteps, prompt_attention_mask = prompt
                prompt_skill_encodes = skill_model.vq_codes_to_encodes(prompt_skills)
                prompt = (prompt_states, prompt_skill_encodes, prompt_timesteps, prompt_attention_mask) if self.model.no_rtg else \
                    (prompt_states, prompt_skill_encodes, prompt_returns_to_go, prompt_timesteps, prompt_attention_mask)
                skills_input = skill_encodes
            else:
                skills_input = skills

        else:
            if self.model.no_rtg:
                states, nxt_states, timesteps, attention_mask = batch
                rtg = None
            else:
                states, nxt_states, rtg, timesteps, attention_mask = batch
            bs, seq_len, _ = states.shape
            skill_states = states.reshape(bs*seq_len, 1, -1)
            skill_nxt_states = nxt_states.reshape(bs*seq_len, 1, -1)
            skills, skill_encodes = skill_model.get_skill_vq_codes_encodes(skill_states, skill_nxt_states)
            skills = skills.reshape(bs, seq_len)
            skill_encodes = skill_encodes.reshape(bs, seq_len, -1)
            
            if self.model.no_rtg:
                prompt_states, prompt_nxt_states, prompt_timesteps, prompt_attention_mask = prompt
            else:
                prompt_states, prompt_nxt_states, prompt_returns_to_go, prompt_timesteps, prompt_attention_mask = prompt
            bs, prompt_seq_len, _ = prompt_states.shape
            skill_prompt_states = prompt_states.reshape(bs*prompt_seq_len, 1, -1)
            skill_prompt_nxt_states = prompt_nxt_states.reshape(bs*prompt_seq_len, 1, -1)
            prompt_skills, prompt_skill_encodes = skill_model.get_skill_vq_codes_encodes(skill_prompt_states, skill_prompt_nxt_states)
            prompt_skills = prompt_skill_encodes.reshape(bs, prompt_seq_len, -1) if self.embed_skill_encode else prompt_skills.reshape(bs, prompt_seq_len)
            prompt = (prompt_states, prompt_skill_encodes, prompt_timesteps, prompt_attention_mask) if self.model.no_rtg else \
                        (prompt_states, prompt_skill_encodes, prompt_returns_to_go, prompt_timesteps, prompt_attention_mask)

            skills_input = skill_encodes if self.embed_skill_encode else skills

        skill_target = torch.clone(skills)

        prompt = None if no_prompt else prompt

        skill_preds = self.model.forward(
            states, skills_input, rtg, timesteps, attention_mask=attention_mask, prompt=prompt
        )

        n_codebook = skill_preds.shape[2]

        skill_preds = skill_preds.reshape(-1, n_codebook)[attention_mask.reshape(-1) > 0]
        skill_target = skill_target.reshape(-1)[attention_mask.reshape(-1) > 0]
        loss = F.cross_entropy(skill_preds, skill_target, reduction='none')
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
        self.optimizer.step()

        skill_error = torch.mean((torch.argmax(skill_preds, dim=-1) != skill_target).float())

        with torch.no_grad():
            self.diagnostics['skill_dt_training/skill_error'] = skill_error.detach().cpu().item()
            self.diagnostics['skill_dt_training/grad_norm'] = grad_norm.detach().cpu().item()
            self.diagnostics['skill_dt_training/learning_rate'] = self.optimizer.state_dict()['param_groups'][0]['lr']
        
        return loss.detach().cpu().item()


    def eval_iteration_multienv(
            self, 
            task_info,
            task_name_list,
            task_env_funcs,
            prompt_trajectories_list,
            variant,
            skill_model,
            iter_num=0,
            print_logs=False,
            group='test',
            vec_env_num=50,
        ):
        print('evaluate at tasks: ', task_name_list)
        logs = dict()
        print('start evaluating...')
        self.model.eval()
        skill_model.eval()

        eval_start = time.time()

        no_prompt = self.config['no_prompt']
        if no_prompt:
            prompt = None
        else:
            get_all_prompt_fn = get_all_prompt(
                prompt_trajectories_list, task_info, task_name_list,
                self.config['stochastic_prompt'], variant['no_state_normalize'],
                keys=self.data_keys, interval_length=self.skill_length
            )
            prompt = get_all_prompt_fn(1, self.prompt_episode, self.prompt_length)

            if self.config['preprocess_skill']:
                if self.embed_skill_encode:
                    if self.model.no_rtg:
                        prompt_states, prompt_skills, prompt_timesteps, prompt_attention_mask = prompt
                    else:
                        prompt_states, prompt_skills, prompt_returns_to_go, prompt_timesteps, prompt_attention_mask = prompt
                    prompt_skill_encodes = skill_model.vq_codes_to_encodes(prompt_skills)
                    prompt = (prompt_states, prompt_skill_encodes, prompt_timesteps, prompt_attention_mask) if self.model.no_rtg else \
                                (prompt_states, prompt_skill_encodes, prompt_returns_to_go, prompt_timesteps, prompt_attention_mask)
            else:
                if self.model.no_rtg:
                    prompt_states, prompt_nxt_states, prompt_timesteps, prompt_attention_mask = prompt
                else:
                    prompt_states, prompt_nxt_states, prompt_returns_to_go, prompt_timesteps, prompt_attention_mask = prompt
                bs, prompt_seq_len, _ = prompt_states.shape
                skill_prompt_states = prompt_states.reshape(bs*prompt_seq_len, 1, -1)
                skill_prompt_nxt_states = prompt_nxt_states.reshape(bs*prompt_seq_len, 1, -1)
                prompt_skills, prompt_skill_encodes = skill_model.get_skill_vq_codes_encodes(skill_prompt_states, skill_prompt_nxt_states)
                prompt_skills = prompt_skill_encodes.reshape(bs, prompt_seq_len, -1) if self.embed_skill_encode else prompt_skills.reshape(bs, prompt_seq_len)
                prompt = (prompt_states, prompt_skill_encodes, prompt_timesteps, prompt_attention_mask) if self.model.no_rtg else \
                            (prompt_states, prompt_skill_encodes, prompt_returns_to_go, prompt_timesteps, prompt_attention_mask)
            

        env_targets = [task_info[env_name]['env_targets'][0] for env_name in task_name_list]
        eval_vec_fn = skill_dt_eval_vec_episodes(env_targets, task_info, variant, task_env_funcs, task_name_list, vec_env_num)
        outputs, skill_id = eval_vec_fn(self.model, skill_model, prompt=prompt)

        for k, v in outputs.items():
            logs[f'{group}-{k}'] = v

        logs['evaluation/time'] = time.time() - eval_start

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        if print_logs:
            print('=' * 80)
            print(f'Iteration {iter_num}')
            for k, v in logs.items():
                print(f'{k}: {v}')

        return logs, skill_id
    

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
            no_state_normalize
            ):
        if self.config['preprocess_skill']:
            self.data_keys = ['observations', 'skills', 'timesteps'] if self.model.no_rtg else \
                                ['observations', 'skills', 'return_to_gos', 'timesteps']
        else:
            self.data_keys = ['observations', 'next_observations', 'timesteps'] if self.model.no_rtg else \
                                ['observations', 'next_observations', 'return_to_gos', 'timesteps']
        self.prompt_batch_fn = get_prompt_batch(
            trajectories_list, prompt_trajectories_list, train_info, train_task_name_list,
            self.config['stochastic_prompt'], no_state_normalize, keys=self.data_keys,
            interval_length=self.skill_length
        )


    def get_model_path(self, env_name, postfix, folder):
        model_name = 'skill_dt_model_' + env_name + postfix
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
    
