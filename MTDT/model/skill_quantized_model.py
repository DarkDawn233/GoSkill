# Code backbone: Decision Transformer https://github.com/kzl/decision-transformer/
# Decision Transformer License: https://github.com/kzl/decision-transformer/blob/master/LICENSE.md

import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers

from .trajectory_gpt2 import GPT2Model

from MTDT.vector_quantize_pytorch import VectorQuantize

class SkillQuantizedTransformer(nn.Module):

    def __init__(
            self,
            src_state_dim,
            state_dim,
            act_dim,
            skill_length=None,
            action_tanh=True,
            config={},
    ):
        super().__init__()
        self.config = config
        self.src_state_dim = src_state_dim
        self.state_dim = state_dim[0]
        self.act_dim = act_dim[0]
        self.skill_length = skill_length
        self.embed_time_step = config['embed_time_step']
        self.multi_head = config['multi_head']
        
        self.n_codebook = config['n_codebook']
        self.vq_size = config['vq_size']

        self.hidden_size = config['hidden_size']
        
        skill_encoder = [nn.Linear(self.state_dim, self.hidden_size), nn.LayerNorm(self.hidden_size)]
        for _ in range(config['mlp_n_layer'] - 1):
            skill_encoder.append(nn.Linear(self.hidden_size, self.hidden_size))
            skill_encoder.append(nn.ReLU())
        skill_encoder.append(nn.Linear(self.hidden_size, self.vq_size))
        skill_encoder.append(nn.Tanh())
        self.skill_encoder = nn.Sequential(*skill_encoder)

        self.skill_vq_layer = VectorQuantize(
            dim=self.vq_size, codebook_size=self.n_codebook, kmeans_init=config['kmeans_init']
        )

        self.action_embed_action = nn.Linear(self.act_dim, self.hidden_size)
        self.action_embed_state = nn.Linear(self.state_dim, self.hidden_size)
        self.action_embed_skill_vq = nn.Linear(self.vq_size, self.hidden_size)

        if self.embed_time_step:
            self.action_embed_timestep = nn.Embedding(self.skill_length, self.hidden_size)

        self.action_encoder_embed_ln = nn.LayerNorm(self.hidden_size)

        action_encoder_n_positions = 2 * self.skill_length + 1
        action_encoder_n_positions += self.skill_length
        action_encoder_config = transformers.GPT2Config(
            vocab_size = 1,
            n_embd = self.hidden_size,
            n_positions = action_encoder_n_positions, # Useless
            n_layer = config['n_layer'],
            n_head = config['n_head'],
            n_inner = self.hidden_size * 4,
            activation_function = config['activation_function'],
            resid_pdrop = config['resid_pdrop'],
            attn_pdrop = config['attn_pdrop']
        )
        self.action_encoder = GPT2Model(action_encoder_config)
        
        if self.multi_head:
            self.predict_action = nn.ModuleList([nn.Sequential(
                *([nn.Linear(self.hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
            ) for _ in range(self.n_codebook)])
        else:
            self.predict_action = nn.Sequential(
                *([nn.Linear(self.hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
            )
    

    def state_skill_encoding(self, states, nxt_states):
        skill_state_change = nxt_states[:, -1] - states[:, 0]
        skill_encode_output = self.skill_encoder(skill_state_change)
        state_change = states[:, :] - states[:, 0].unsqueeze(1)
        state_encode_output = self.skill_encoder(state_change)

        return skill_encode_output, state_encode_output


    def skill_quantize(self, skill_encodes, freeze_codebook = False):
        encodes = skill_encodes.unsqueeze(1)
        sample_codebook_temp = 0 if freeze_codebook else None
        vq_encodes, vq_codes, vq_loss = self.skill_vq_layer(
            encodes, freeze_codebook=freeze_codebook, sample_codebook_temp=sample_codebook_temp
        )
        skill_vq_encodes = vq_encodes[:, 0]
        skill_vq_codes = vq_codes[:, 0]

        return skill_vq_encodes, skill_vq_codes, vq_loss


    def action_predict(self, skill_vq_encodes, skill_vq_codes, state_vq_encodes, states, actions, attention_mask):
        batch_size, skill_length = actions.shape[0], actions.shape[1]
        assert skill_length == self.skill_length

        skill_vq_encode_embeddings = self.action_embed_skill_vq(skill_vq_encodes)
        state_embeddings = self.action_embed_state(states)
        action_embeddings = self.action_embed_action(actions)
        state_vq_encodes = self.action_embed_skill_vq(state_vq_encodes)

        if self.embed_time_step:
            timestep_embeddings = self.action_embed_timestep(torch.arange(self.skill_length, device=actions.device))
            timestep_embeddings = timestep_embeddings.unsqueeze(0).expand(batch_size, -1, -1)

            state_embeddings = state_embeddings + timestep_embeddings
            action_embeddings = action_embeddings + timestep_embeddings
            state_vq_encodes = state_vq_encodes + timestep_embeddings

        stacked_encoder_inputs = torch.stack(
            (state_vq_encodes, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, skill_length * 3, self.hidden_size)
        
        stacked_encoder_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, skill_length * 3)

        
        stacked_encoder_inputs = torch.cat(
            (skill_vq_encode_embeddings, stacked_encoder_inputs), dim=1
        )

        stacked_encoder_attention_mask = torch.cat(
            (torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=attention_mask.device), stacked_encoder_attention_mask), dim=1
        )

        stacked_encoder_inputs = self.action_encoder_embed_ln(stacked_encoder_inputs)

        encoder_outputs = self.action_encoder(
            inputs_embeds=stacked_encoder_inputs,
            attention_mask=stacked_encoder_attention_mask,
        )

        x = encoder_outputs['last_hidden_state'][:, 1:]
        x = x.reshape(batch_size, skill_length, -1, self.hidden_size).permute(0, 2, 1, 3)
        encode_output = x[:, -2]

        if self.multi_head:
            action_preds = [predict_action(encode_output) for predict_action in self.predict_action]
            action_preds = torch.stack(action_preds, dim=-1)
            skill_onehots = F.one_hot(skill_vq_codes, num_classes=self.n_codebook).float()
            action_preds = torch.sum(action_preds * skill_onehots.unsqueeze(1), dim=-1)
        else:
            action_preds = self.predict_action(encode_output)

        return action_preds
    

    def forward(self, states, nxt_states, actions, attention_mask = None, freeze_codebook = False):
        batch_size, seq_length = states.shape[0], states.shape[1]

        assert seq_length == self.skill_length

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.float32, device=states.device)

        skill_encodes, state_encodes = self.state_skill_encoding(states, nxt_states)
        
        if state_encodes is not None:
            all_encodes = torch.cat([state_encodes, skill_encodes.unsqueeze(1)], dim=1)
        else:
            all_encodes = skill_encodes.unsqueeze(1)

        skill_vq_encodes, skill_vq_codes, vq_loss = \
            self.skill_quantize(skill_encodes, freeze_codebook = freeze_codebook)
        state_vq_encodes = state_encodes

        expand_skill_vq_encodes = skill_vq_encodes.reshape(batch_size, 1, self.vq_size)
        expand_skill_vq_codes = skill_vq_codes.reshape(batch_size, 1)
        if freeze_codebook:
            expand_skill_vq_encodes = expand_skill_vq_encodes.detach()
            expand_skill_vq_codes = expand_skill_vq_codes.detach()
            state_vq_encodes = state_vq_encodes.detach() if state_vq_encodes is not None else None

        action_pre = self.action_predict(expand_skill_vq_encodes, expand_skill_vq_codes, state_vq_encodes,
                                         states, actions, attention_mask)

        skill_vq_codes = skill_vq_codes.reshape(batch_size)

        return action_pre, skill_vq_codes, vq_loss, all_encodes
    

    def get_skill_vq_codes_encodes(self, states, nxt_states):
        batch_size, seq_length = states.shape[0], states.shape[1]
        assert seq_length in [self.skill_length, 1]
        skill_encodes, _ = self.state_skill_encoding(states, nxt_states)

        skill_vq_encodes, skill_vq_codes, _ = self.skill_quantize(skill_encodes, freeze_codebook = True)

        skill_vq_codes = skill_vq_codes.reshape(batch_size)
        skill_vq_encodes = skill_vq_encodes.reshape(batch_size, self.vq_size)
        
        return skill_vq_codes, skill_vq_encodes
    
    
    def vq_codes_to_encodes(self, vq_codes):
        if vq_codes.dim() == 2:
            skill_vq_encodes = self.skill_vq_layer.get_codes_from_indices(vq_codes)
        elif vq_codes.dim() == 1:
            skill_vq_encodes = self.skill_vq_layer.get_codes_from_indices(vq_codes.unsqueeze(1))
            skill_vq_encodes = skill_vq_encodes.squeeze(1)
        else:
            raise ValueError(f"Error dim for vq_code2encode: {vq_codes.dim()}")
        return skill_vq_encodes


    def get_action(self, states, actions, skills):
        env_num = states.shape[0]
        skill_step = states.shape[1] - 1

        attention_mask = torch.cat([torch.ones(states.shape[1]), torch.zeros(self.skill_length-states.shape[1])])
        attention_mask = attention_mask.to(dtype=torch.float32, device=states.device).reshape(1, -1)
        attention_mask = attention_mask.repeat(env_num, 1)

        states = torch.cat(
            [states, torch.zeros((states.shape[0], self.skill_length-states.shape[1], states.shape[-1]), device=states.device, dtype=torch.float32)],
            dim=1)
        actions = torch.cat(
            [actions, torch.zeros((actions.shape[0], self.skill_length-actions.shape[1], actions.shape[-1]), device=actions.device, dtype=torch.float32)],
            dim=1)
        
        state_change = states[:, :] - states[:, 0].unsqueeze(1)
        state_encodes = self.skill_encoder(state_change)
        state_vq_encodes = state_encodes
        
        skill_encodes = self.vq_codes_to_encodes(skills)

        action_preds = self.action_predict(skill_encodes, skills, state_vq_encodes,
                                           states, actions, attention_mask)

        return action_preds[:, skill_step]
