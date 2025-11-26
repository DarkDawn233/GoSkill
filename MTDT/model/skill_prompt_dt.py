# Code backbone: Decision Transformer https://github.com/kzl/decision-transformer/
# Decision Transformer License: https://github.com/kzl/decision-transformer/blob/master/LICENSE.md

import torch
import torch.nn as nn

import transformers

from .trajectory_gpt2 import GPT2Model

class SkillPromptDecisionTransformer(nn.Module):

    def __init__(
            self,
            state_dim,
            act_dim,
            skill_length,
            vq_size,
            n_codebook,
            no_rtg=False,
            config={}
    ):
        super().__init__()
        self.state_dim = state_dim[0]
        self.act_dim = act_dim[0]
        self.config = config
        self.no_rtg = no_rtg
        self.max_length = config['max_length']
        self.max_ep_len = config['max_ep_len']
        self.hidden_size = config['hidden_size']
        self.skill_length = skill_length
        self.vq_size = vq_size
        self.n_codebook = n_codebook

        self.embed_skill_encode = config['embed_skill_encode']

        config = transformers.GPT2Config(
            vocab_size = 1,
            n_embd = config['hidden_size'],
            n_positions = config['n_positions'], # Useless
            n_layer = config['n_layer'],
            n_head = config['n_head'],
            n_inner = self.hidden_size * 4,
            activation_function = config['activation_function'],
            resid_pdrop = config['resid_pdrop'],
            attn_pdrop = config['attn_pdrop'],
        )

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(config)
        # change to parallelize mode for metaworld big model
        # self.transformer.parallelize()

        self.embed_timestep = nn.Embedding(self.max_ep_len, self.hidden_size)
        if not self.no_rtg:
            self.embed_return = nn.Linear(1, self.hidden_size)
        self.embed_state = nn.Linear(self.state_dim, self.hidden_size)
        if self.embed_skill_encode:
            self.embed_skill = nn.Linear(self.vq_size, self.hidden_size)
        else:
            self.embed_skill = nn.Embedding(self.n_codebook, self.hidden_size)

        self.prompt_embed_timestep = nn.Embedding(self.max_ep_len, self.hidden_size)
        if not self.no_rtg:
            self.prompt_embed_return = nn.Linear(1, self.hidden_size)
        self.prompt_embed_state = nn.Linear(self.state_dim, self.hidden_size)
        if self.embed_skill_encode:
            self.prompt_embed_skill = nn.Linear(self.vq_size, self.hidden_size)
        else:
            self.prompt_embed_skill = nn.Embedding(self.n_codebook, self.hidden_size)

        self.embed_ln = nn.LayerNorm(self.hidden_size)

        self.predict_skill = nn.Linear(self.hidden_size, self.n_codebook)
    

    def forward(self, states, skills, returns_to_go, timesteps, attention_mask=None, prompt=None):
        batch_size, seq_length = states.shape[0], states.shape[1]
        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.float32, device=states.device)

        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        skill_embeddings = self.embed_skill(skills)
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        skill_embeddings = skill_embeddings + time_embeddings
        inputs_list = [state_embeddings, skill_embeddings]

        if not self.no_rtg:
            returns_embeddings = self.embed_return(returns_to_go)
            returns_embeddings = returns_embeddings + time_embeddings
            inputs_list.insert(0, returns_embeddings)

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = torch.stack(
            inputs_list, dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, len(inputs_list) * seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = torch.stack(
            [attention_mask] * len(inputs_list), dim=1
        ).permute(0, 2, 1).reshape(batch_size, len(inputs_list) * seq_length)

        # process prompt the same as d-t
        if prompt is not None:
            if self.no_rtg:
                prompt_states, prompt_skills, prompt_timesteps, prompt_attention_mask = prompt
            else:
                prompt_states, prompt_skills, prompt_returns_to_go, prompt_timesteps, prompt_attention_mask = prompt
            prompt_seq_length = prompt_states.shape[1]

            prompt_state_embeddings = self.prompt_embed_state(prompt_states)
            prompt_skill_embeddings = self.prompt_embed_skill(prompt_skills)
            prompt_time_embeddings = self.prompt_embed_timestep(prompt_timesteps)

            prompt_state_embeddings = prompt_state_embeddings + prompt_time_embeddings
            prompt_skill_embeddings = prompt_skill_embeddings + prompt_time_embeddings
            prompt_inputs_list = [prompt_state_embeddings, prompt_skill_embeddings]
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
        skill_preds = self.predict_skill(x[:, -2])    # predict next skill given state and action
        
        return skill_preds

    
    def reset_eval(self):
        self.skill_cnt = 0
        self.last_skill = None


    def get_skill(self, states, skills, skill_encodes, returns_to_go, timesteps, prompt):
        env_num = states.shape[0]

        if self.max_length is not None:
            states = states[:,-self.max_length:]
            skills = skills[:,-self.max_length:]
            skill_encodes = skill_encodes[:,-self.max_length:]
            returns_to_go = returns_to_go[:,-self.max_length:]
            timesteps = timesteps[:,-self.max_length:]

            # pad all tokens to sequence length
            attention_mask = torch.cat([torch.zeros(self.max_length-states.shape[1]), torch.ones(states.shape[1])])
            attention_mask = attention_mask.to(dtype=torch.float32, device=states.device).reshape(1, -1)
            attention_mask = attention_mask.repeat(env_num, 1)
            
            states = torch.cat(
                [torch.zeros((states.shape[0], self.max_length-states.shape[1], states.shape[-1]), device=states.device, dtype=torch.float32),
                 states], dim=1)
            skills = torch.cat(
                [torch.zeros((skills.shape[0], self.max_length-skills.shape[1]), device=skills.device, dtype=torch.long),
                 skills], dim=1)
            skill_encodes = torch.cat(
                [torch.zeros((skill_encodes.shape[0], self.max_length-skill_encodes.shape[1], self.vq_size), device=skill_encodes.device, dtype=torch.float32),
                 skill_encodes], dim=1)
            returns_to_go = torch.cat(
                [torch.zeros((returns_to_go.shape[0], self.max_length-returns_to_go.shape[1], 1), device=returns_to_go.device, dtype=torch.float32), 
                 returns_to_go], dim=1)
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1]), device=timesteps.device, dtype=torch.long),
                 timesteps], dim=1)
        else:
            attention_mask = None

        skills_input = skill_encodes if self.embed_skill_encode else skills
        
        skill_preds = self.forward(states, skills_input, returns_to_go, timesteps, attention_mask=attention_mask, prompt=prompt)

        skill_code_preds = torch.argmax(skill_preds[:, -1], dim=-1)
        return skill_code_preds
