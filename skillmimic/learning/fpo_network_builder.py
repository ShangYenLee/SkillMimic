import torch, math
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from rl_games.algos_torch import network_builder
from rl_games.algos_torch.models import BaseModel
from rl_games.algos_torch.network_builder import NetworkBuilder, A2CBuilder

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

class adaLN(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = self.norm1(x) * (1+scale) + shift
        return x
    
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class FlowMatchingOptimBuilder(A2CBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs) 

    def load(self, params):
        self.separate = params.get('separate', False)
        self.units = params['mlp']['units']
        self.activation = params['mlp']['activation']
        self.initializer = params['mlp']['initializer']
        self.is_d2rl = params['mlp'].get('d2rl', False)
        self.norm_only_first_layer = params['mlp'].get('norm_only_first_layer', False)
        self.value_activation = params.get('value_activation', 'None')
        self.normalization = params.get('normalization', None)
        self.has_rnn = 'rnn' in params
        self.has_space = 'space' in params
        self.central_value = params.get('central_value', False)
        self.joint_obs_actions_config = params.get('joint_obs_actions', None)
        self.params = params

    class Network(A2CBuilder.Network):
        def __init__(self, params, prior_noise_std=1.0, **kwargs):
            super().__init__(params, **kwargs)
            #NetworkBuilder.BaseNetwork.__init__(self)
            self.load(params)
            input_shape = kwargs.pop('input_shape')[0]
            self.action_size = kwargs.pop('actions_num')
            self.value_size = kwargs.pop('value_size', 1)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.register_buffer("prior_noise_std", torch.tensor(prior_noise_std, device=device))
            #self.prior_noise_std = torch.tensor(prior_noise_std)
            self.p_mean = -1.2
            self.p_std = 1.2
            self.actor_mlp = nn.Sequential()
            self.critic_mlp = nn.Sequential()

            actor_mlp_args = {
                'input_size': input_shape + self.action_size,
                'units': self.units,
                'activation': self.activation,
                'norm_func_name': self.normalization,
                'dense_func': torch.nn.Linear,
                'd2rl': self.is_d2rl,
                'norm_only_first_layer': self.norm_only_first_layer
            }
            critic_mlp_args = {
                'input_size': input_shape,
                'units': self.units,
                'activation': self.activation,
                'norm_func_name': self.normalization,
                'dense_func': torch.nn.Linear,
                'd2rl': self.is_d2rl,
                'norm_only_first_layer': self.norm_only_first_layer
            }
            # cnn_args = {
            #     'ctype': self.cnn['type'],
            #     'input_shape': input_shape,
            #     'convs': self.cnn['convs'],
            #     'activation': self.cnn['activation'],
            #     'norm_func_name': self.normalization,
            # }
            out_size = self.units[-1]
            self.actor_mlp = self._build_mlp(**actor_mlp_args)
            self.critic_mlp = self._build_mlp(**critic_mlp_args)
            self.actor_norm = adaLN(out_size)
            self.mu = nn.Sequential(
                layer_init(nn.Linear(out_size, self.action_size), std=0.01),
            )
            self.post_adaln_non_linearity = nn.SiLU()
            self.noise_emb = TimestepEmbedder(out_size)
            self.value = nn.Linear(out_size, self.value_size)
            self.value_act = nn.Identity()
            # self.value = self._build_value_layer(out_size, self.value_size)
            # self.value_act = self.activations_factory.create(self.value_activation)
            # self.critic_cnn = self._build_conv(**cnn_args)

        def sample_noise(self, noise_shape, device):
            noise = torch.randn(noise_shape, dtype=torch.float32, device=device)
            noise = noise * self.prior_noise_std

            return noise
        
        def sample_ts(self, B, device):
            rnd_normal = torch.randn((B,), device=device)
            sigma = (rnd_normal * self.p_std + self.p_mean).exp()
            time = 1 / (1 + sigma)
            time = torch.clip(time, min=0.0001, max=1.0)
            return time #torch.rand(B, device=device)

        def forward(self, obs_dict, noise=None, t=None):
            obs = obs_dict['obs']
            action = obs_dict['actions'].to(obs.device)
            states = obs_dict.get('rnn_states', None)
            dones = obs_dict.get('dones', None)

            noise = self.sample_noise(action.shape, action.device) if noise is None else noise
            t = self.sample_ts(action.shape[0], action.device) if t is None else t
            x_t = (1 - t).unsqueeze(1) * noise + t.unsqueeze(1) * action
            u_t = action - noise 
            velocity = self.eval_actor(obs, x_t, t)
            value = self.eval_critic(obs)
            log_probs = - ((u_t - velocity) ** 2) / (2 * 0.5 **2)
            flow_matching_loss = - log_probs.reshape(-1).mean()

            # w = (1 - t) + 1e-6
            # lambda_dt = torch.ones_like(t)
            # weight = 0.5 * w * lambda_dt
            # flow_matching_loss = (weight * (velocity - u_t).pow(2).mean(dim=-1)).mean()
            
            # flow_matching_loss = F.mse_loss(velocity, u_t, reduction='none').mean(dim=-1)

            return {
                'pred_velocity': velocity,
                'target_velocity': u_t,
                'flow_matching_loss':flow_matching_loss,
                'x_t': x_t,
                't': t,
                'values': value,
                'states': states,
                'entropy': torch.zeros_like(value)
            }

        def eval_actor(self, obs, x_t=None, t=None):
            B = obs.shape[0]
            # device = obs.device
            # if x_t is None:
            #     x_t = torch.zeros_like(obs[:, :self.action_dim], device=device)
            # if t is None:
            #     t = torch.rand(B, device=device)

            # t_embed = self.timestep_embedding
            # t_embed = t_embed.unsqueeze(1)

            if obs.dim() > 2:
                obs = obs.view(B, -1)
            t_batch = torch.ones([B], device=obs.device) * t
            noise_emb = self.noise_emb(t_batch * 1.0)
            x_in = torch.cat([x_t, obs], dim=-1)
            hidden = self.actor_mlp(x_in)
            hidden = self.actor_norm(hidden, noise_emb)
            hidden = self.post_adaln_non_linearity(hidden)
            velocity = self.mu(hidden)

            return velocity
        
        def eval_critic(self, obs):
            c_out = obs
            c_out = c_out.contiguous().view(c_out.size(0), -1)
            c_out = self.critic_mlp(c_out)              
            value = self.value_act(self.value(c_out))
            return value
        
        def sample_action(self, obs, num_steps=100):
            """
            Flow-based action sampling via Euler integration
            """
            self.eval()
            device = obs.device
            batch_size = obs.shape[0]
            x_t = torch.randn(batch_size, self.action_size, device=device)
            dt = 1.0 / num_steps

            with torch.no_grad():
                for i in range(num_steps):
                    t = torch.full((batch_size,), (i + 0.5)/num_steps, device=device)
                    # t = torch.full((batch_size,), i / num_steps, device=device)
                    v_t = self.eval_actor(obs, x_t, t)
                    x_t = x_t + dt * v_t
            return x_t.clamp(-1.0, 1.0)

    def build(self, name, **kwargs):
        net = FlowMatchingOptimBuilder.Network(self.params, **kwargs)
        return net
