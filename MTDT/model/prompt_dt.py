# Code backbone: Decision Transformer https://github.com/kzl/decision-transformer/
# Decision Transformer License: https://github.com/kzl/decision-transformer/blob/master/LICENSE.md

import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers

from .trajectory_gpt2 import GPT2Model

class PromptDecisionTransformer(nn.Module):

    def __init__(
            self,
            src_state_dim,
            state_dim,
            act_dim,
            task_num,
            no_rtg=False,
            action_tanh=True,
            config={},
    ):
        super().__init__()
        self.src_state_dim = src_state_dim
        self.state_dim = state_dim[0]
        self.act_dim = act_dim[0]
        self.task_num = task_num
        self.config = config
        self.no_rtg = no_rtg
        self.multi_head = config['multi_head']
        self.max_length = config['max_length']
        self.max_ep_len = config['max_ep_len']
        self.hidden_size = config['hidden_size']
        self.task_input = config['task_input']

        gpt2_config = transformers.GPT2Config(
            vocab_size = 1,
            n_embd = config['hidden_size'],
            n_layer = config['n_layer'],
            n_head = config['n_head'],
            n_inner = self.hidden_size * 4,
            activation_function = config['activation_function'],
            n_positions = config['n_positions'],
            resid_pdrop = config['resid_pdrop'],
            attn_pdrop = config['attn_pdrop'],
        )

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(gpt2_config)
        # change to parallelize mode for metaworld big model
        # self.transformer.parallelize()

        self.embed_timestep = nn.Embedding(self.max_ep_len, self.hidden_size)
        if not self.no_rtg:
            self.embed_return = torch.nn.Linear(1, self.hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, self.hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, self.hidden_size)

        self.prompt_embed_timestep = nn.Embedding(self.max_ep_len, self.hidden_size)
        if not self.no_rtg:
            self.prompt_embed_return = torch.nn.Linear(1, self.hidden_size)
        self.prompt_embed_state = torch.nn.Linear(self.state_dim, self.hidden_size)
        self.prompt_embed_action = torch.nn.Linear(self.act_dim, self.hidden_size)

        self.embed_ln = nn.LayerNorm(self.hidden_size)

        # note: we don't predict states or returns for the paper
        # self.predict_state = torch.nn.Linear(self.hidden_size, self.state_dim)

        if self.multi_head:
            self.predict_action = nn.ModuleList([nn.Sequential(
                *([nn.Linear(self.hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
            ) for _ in range(task_num)])
        else:
            self.predict_action = nn.Sequential(
                *([nn.Linear(self.hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
            )
        # self.predict_return = torch.nn.Linear(self.hidden_size, 1)


    def forward(self, states, actions, returns_to_go, timesteps, task_ids=None, attention_mask=None, prompt=None):
        batch_size, seq_length = states.shape[0], states.shape[1]
        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.float32)

        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)

        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        inputs_list = [state_embeddings, action_embeddings]

        if not self.no_rtg:
            returns_embeddings = self.embed_return(returns_to_go)
            returns_embeddings = returns_embeddings + time_embeddings
            inputs_list.insert(0, returns_embeddings)

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = torch.stack(
            inputs_list, dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, len(inputs_list)*seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = torch.stack(
            [attention_mask] * len(inputs_list), dim=1
        ).permute(0, 2, 1).reshape(batch_size, len(inputs_list)*seq_length)

        # process prompt the same as d-t
        if prompt is not None:
            if self.no_rtg:
                prompt_states, prompt_actions, prompt_timesteps, _, prompt_attention_mask = prompt
            else:
                prompt_states, prompt_actions, prompt_returns_to_go, prompt_timesteps, _, prompt_attention_mask = prompt
            prompt_seq_length = prompt_states.shape[1]
            prompt_state_embeddings = self.prompt_embed_state(prompt_states)
            prompt_action_embeddings = self.prompt_embed_action(prompt_actions)
            prompt_time_embeddings = self.prompt_embed_timestep(prompt_timesteps)
            
            prompt_state_embeddings = prompt_state_embeddings + prompt_time_embeddings
            prompt_action_embeddings = prompt_action_embeddings + prompt_time_embeddings
            prompt_inputs_list = [prompt_state_embeddings, prompt_action_embeddings]
            if not self.no_rtg:
                prompt_returns_embeddings = self.prompt_embed_return(prompt_returns_to_go)
                prompt_returns_embeddings = prompt_returns_embeddings + prompt_time_embeddings
                prompt_inputs_list.insert(0, prompt_returns_embeddings)

            prompt_stacked_inputs = torch.stack(
                prompt_inputs_list, dim=1
            ).permute(0, 2, 1, 3).reshape(prompt_states.shape[0], len(prompt_inputs_list) * prompt_seq_length, self.hidden_size)

            # to make the attention mask fit the stacked inputs, have to stack it as well
            prompt_stacked_attention_mask = torch.stack(
                [prompt_attention_mask] * len(prompt_inputs_list), dim=1
            ).permute(0, 2, 1).reshape(prompt_states.shape[0], len(prompt_inputs_list) * prompt_seq_length)

            # stacked_inputs add prompted sequence
            stacked_inputs = torch.cat((prompt_stacked_inputs, stacked_inputs), dim=1)
            stacked_attention_mask = torch.cat((prompt_stacked_attention_mask, stacked_attention_mask), dim=1)
        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state'][:, -seq_length * len(inputs_list):, :]

        x = x.reshape(batch_size, -1, len(inputs_list), self.hidden_size).permute(0, 2, 1, 3)

        # note here all the prompt are pre-append to x, but when return only return the last [:, -seq_length:, :] corresponding to batch data
        # get predictions
        # return_preds = self.predict_return(x[:,2])[:, -seq_length:, :]  # predict next return given state and action
        # state_preds = self.predict_state(x[:,2])[:, -seq_length:, :]    # predict next state given state and action
        if self.multi_head:
            action_preds = [head(x[:, -2]) for head in self.predict_action]
            action_preds = torch.stack(action_preds, dim=1)  
            task_ids = F.one_hot(task_ids, num_classes=self.task_num).to(dtype=torch.float32) 
            action_preds = torch.sum(action_preds * task_ids.unsqueeze(2).unsqueeze(3), dim=1) 
        else:
            action_preds = self.predict_action(x[:, -2])  # predict next action given state

        return action_preds

    def get_action(self, states, actions, rewards, returns_to_go, timesteps, task_ids=None, **kwargs):
        # we don't care about the past rewards in this model
        # Parralle env as batch
        env_num = states.shape[0]

        if self.max_length is not None:
            states = states[:,-self.max_length:]
            actions = actions[:,-self.max_length:]
            returns_to_go = returns_to_go[:,-self.max_length:]
            timesteps = timesteps[:,-self.max_length:]

            # pad all tokens to sequence length
            attention_mask = torch.cat([torch.zeros(self.max_length-states.shape[1]), torch.ones(states.shape[1])])
            attention_mask = attention_mask.to(dtype=torch.float32, device=states.device).reshape(1, -1)
            attention_mask = attention_mask.repeat(env_num, 1)
            states = torch.cat(
                [torch.zeros((states.shape[0], self.max_length-states.shape[1], states.shape[-1]), device=states.device), states],
                dim=1).to(dtype=torch.float32)
            actions = torch.cat(
                [torch.zeros((actions.shape[0], self.max_length - actions.shape[1], actions.shape[-1]),
                             device=actions.device), actions],
                dim=1).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [torch.zeros((returns_to_go.shape[0], self.max_length-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
                dim=1).to(dtype=torch.float32)
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1]), device=timesteps.device), timesteps],
                dim=1
            ).to(dtype=torch.long)
        else:
            attention_mask = None

        # Note: prompt within kwargs
        action_preds = self.forward(
            states, actions, returns_to_go, timesteps, task_ids=task_ids, attention_mask=attention_mask, **kwargs)

        return action_preds[:,-1]

