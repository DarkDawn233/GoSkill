# Code backbone: Decision Transformer https://github.com/kzl/decision-transformer/
# Decision Transformer License: https://github.com/kzl/decision-transformer/blob/master/LICENSE.md

import numpy as np
import torch
import time

from MTDT.utils.prompt_fns import get_prompt_batch, get_all_prompt
from MTDT.evaluate.prompt_evaluation_fn import eval_vec_episodes


class PromptSequenceTrainer:

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
        self.max_len = self.config['max_length']
        self.prompt_episode = self.config['prompt_episode']
        self.prompt_length = self.config['prompt_length']
        self.diagnostics = dict()

        self.start_time = time.time()


    def pure_train_iteration_mix(self):

        train_losses = []
        logs = dict()

        train_start = time.time()

        self.model.train()
        train_loss = self.train_step_mix()
        train_losses.append(train_loss)
        if self.scheduler is not None:
            self.scheduler.step()

        logs['training/time'] = time.time() - train_start
        logs['training/train_loss_mean'] = np.mean(train_losses)

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        return logs


    def train_step_mix(self):
        no_prompt = self.config['no_prompt']
        prompt, batch = self.prompt_batch_fn(
            batch_size=self.batch_size,
            max_len=self.max_len,
            prompt_episode=self.prompt_episode,
            prompt_length=self.prompt_length,
            no_prompt=no_prompt,
            l_blank=self.max_len-1, r_blank=0,
        )
        if self.model.no_rtg:
            states, actions, timesteps, task_ids, attention_mask = batch
            rtg = None
        else:
            states, actions, rtg, timesteps, task_ids, attention_mask = batch
        action_target = torch.clone(actions)
        prompt = None if no_prompt else prompt
        action_preds = self.model.forward(
            states, actions, rtg, timesteps, task_ids=task_ids,
            attention_mask=attention_mask, prompt=prompt
        )

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        loss = torch.mean((action_preds-action_target)**2)

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/action_error'] = torch.mean((action_preds-action_target)**2).detach().cpu().item()
            self.diagnostics['training/grad_norm'] = grad_norm.detach().cpu().item()
            self.diagnostics['training/learning_rate'] = self.optimizer.state_dict()['param_groups'][0]['lr']
        return loss.detach().cpu().item()
    

    def eval_iteration_multienv(
            self,
            task_info,
            task_name_list,
            task_env_funcs,
            prompt_trajectories_list,
            variant,
            iter_num=0,
            print_logs=False,
            group='test',
            vec_env_num=50,
        ):
        print('evaluate at tasks: ', task_name_list)
        logs = dict()
        print('start evaluating...')
        self.model.eval()

        eval_start = time.time()

        no_prompt = self.config['no_prompt']
        if no_prompt:
            prompt = None
        else:
            get_all_prompt_fn = get_all_prompt(
                prompt_trajectories_list, task_info, task_name_list,
                self.config['stochastic_prompt'],
                variant['no_state_normalize'], keys=self.data_keys
            )
            prompt = get_all_prompt_fn(1, self.prompt_episode, self.prompt_length)
        
        env_targets = [task_info[env_name]['env_targets'][0] for env_name in task_name_list]
        eval_vec_fn = eval_vec_episodes(env_targets, task_info, variant, task_env_funcs, task_name_list, vec_env_num)
        outputs = eval_vec_fn(self.model, prompt=prompt)

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

        return logs


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
            ):
        if self.model.no_rtg:
            self.data_keys = ['observations', 'actions', 'timesteps', 'task_index']
        else:
            self.data_keys = ['observations', 'actions', 'return_to_gos', 'timesteps', 'task_index']
        self.prompt_batch_fn = get_prompt_batch(
            trajectories_list, prompt_trajectories_list, train_info, train_task_name_list,
            self.config['stochastic_prompt'], no_state_normalize, keys=self.data_keys
        )
    

    def get_model_path(self, env_name, postfix, folder):
        model_name = 'prompt_model_' + env_name + postfix
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

