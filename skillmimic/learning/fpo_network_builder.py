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
            nn.ReLU(),
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
        self.num_t = params.get('num_t')
        self.units = params['mlp']['units']
        self.activation = params['mlp']['activation']
        self.initializer = params['mlp']['initializer']
        self.is_d2rl = params['mlp'].get('d2rl', False)
        self.norm_only_first_layer = params['mlp'].get('norm_only_first_layer', False)
        self.value_activation = params.get('value_activation', 'None')
        self.normalization = params.get('normalization', 'layer_norm')
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
            self.num_t = params.get("num_t", 4)
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
            out_size = self.units[-1]
            self.actor_mlp = self._build_mlp(**actor_mlp_args)
            self.critic_mlp = self._build_mlp(**critic_mlp_args)
            # self.actor_mlp = self._build_simple_mlp(input_shape + self.action_size, self.units)
            # self.critic_mlp = self._build_simple_mlp(input_shape, self.units)
            self.actor_norm = adaLN(out_size)
            self.mu = nn.Sequential(
               layer_init(nn.Linear(out_size, self.action_size), std=1.0),
            )
            # self.mu = nn.Sequential(
            #     nn.LayerNorm(out_size, elementwise_affine=False, eps=1e-6),
            #     nn.Linear(out_size, self.action_size)   
            # )
            # self.mu = nn.Linear(out_size, self.action_size)
            self.post_adaln_non_linearity = nn.ReLU()
            self.noise_emb = TimestepEmbedder(out_size)
            # self.value = nn.Linear(out_size, self.value_size)
            self.value = nn.Sequential(
               layer_init(nn.Linear(out_size, self.value_size), std=1.0),
            )
            # self.value = nn.Sequential(
            #     nn.LayerNorm(out_size, elementwise_affine=False, eps=1e-6),
            #     nn.Linear(out_size, self.value_size)   
            # )
            # self.value_act = self.activations_factory.create(self.value_activation)
            #self.value_act = nn.Identity()
            # self.value = self._build_value_layer(out_size, self.value_size)
            # self.value_act = self.activations_factory.create(self.value_activation)
            # self.critic_cnn = self._build_conv(**cnn_args)
            # nn.init.constant_(self.actor_norm.adaLN_modulation[-1].weight, 0)
            # nn.init.constant_(self.actor_norm.adaLN_modulation[-1].bias, 0)

        # def _build_simple_mlp(self, input_size, hidden_sizes, use_layer_norm=True):
        #     layers = []
        #     in_dim = input_size
        #     for h in hidden_sizes:
        #         layers.append(nn.Linear(in_dim, h))
        #         nn.init.kaiming_normal_(layers[-1].weight)
        #         nn.init.zeros_(layers[-1].bias)

        #         if use_layer_norm:
        #             layers.append(nn.LayerNorm(h, eps=1e-6))
        #         layers.append(nn.SiLU())   
        #         in_dim = h
        #     return nn.Sequential(*layers)

        def _build_simple_mlp(self, input_size, hidden_sizes, use_layer_norm=True):
            layers = []
            in_dim = input_size
            for h in hidden_sizes:
                lin = nn.Linear(in_dim, h)
                # Xavier 對 SiLU 比較穩
                nn.init.xavier_uniform_(lin.weight)
                nn.init.zeros_(lin.bias)
                layers.append(lin)

                if use_layer_norm:
                    layers.append(nn.LayerNorm(h, eps=1e-6))
                layers.append(nn.SiLU())
                in_dim = h
            return nn.Sequential(*layers)

        def sample_noise(self, noise_shape, device):
            noise = torch.randn(noise_shape, dtype=torch.float32, device=device)
            noise = noise * self.prior_noise_std

            return noise
        
        def check(self, name, x):
            if not torch.isfinite(x).all():
                print(f"[NaN DETECTED] at {name}")
                print("shape:", x.shape)
                print("stats:", x.min(), x.max(), x.mean())
                raise ValueError(f"NaN found in {name}")
            
        def check_params(self, model):
            for n, p in model.named_parameters():
                if torch.isnan(p).any():
                    print("FOUND NAN WEIGHT:", n)
                    return True
            return False
            
        def sample_ts(self, B, device):
            rnd_normal = torch.randn((B,), device=device)
            # sigma = (rnd_normal * self.p_std + self.p_mean).exp()
            # time = 1 / (1 + sigma)
            # time = torch.clip(time, min=0.0001, max=1.0)
            # return time #torch.rand(B, device=device)
            return rnd_normal

        def forward(self, obs_dict, noise=None, t=None):
            obs = obs_dict['obs']
            action = obs_dict['actions'].to(obs.device)
            states = obs_dict.get('rnn_states', None)
            dones = obs_dict.get('dones', None)
            B = action.shape[0]
            self.check("obs", obs)
            self.check("action", action)

            flat_obs = obs.unsqueeze(1).repeat(1, self.num_t, 1).reshape(B * self.num_t, -1)
            self.check("flat_obs", flat_obs)
            flat_acts = action.unsqueeze(1).expand(B, self.num_t, self.action_size).reshape(B * self.num_t, -1)
            self.check("flat_acts", flat_acts)
            noise = self.sample_noise(flat_acts.shape, action.device) if noise is None else noise
            self.check("noise", noise)
            t = self.sample_ts(flat_acts.shape[0], action.device) if t is None else t
            self.check("t", t)
            x_t = (1 - t).unsqueeze(1) * noise + t.unsqueeze(1) * flat_acts
            self.check("x_t", x_t)
            u_t = flat_acts - noise 
            self.check("u_t", u_t)
            velocity = self.eval_actor(flat_obs, x_t, t)
            self.check("velocity", velocity)
            value = self.eval_critic(obs)
            self.check("value", value)

            # w = (1 - t) + 1e-6
            # lambda_dt = torch.ones_like(t)
            # weight = 0.5 * w * lambda_dt
            # flow_matching_loss = (weight * (velocity - u_t).pow(2).mean(dim=-1))
            flow_matching_loss = nn.functional.mse_loss(velocity, u_t, reduction="none").mean(dim=1)

            flow_matching_loss = flow_matching_loss.view(B, self.num_t).mean(dim=1)
            self.check("flow_matching_loss_pre_reshape", flow_matching_loss)

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

            if obs.dim() > 2:
                obs = obs.view(B, -1)
            # t_batch = torch.ones([B], device=obs.device) * t
            # noise_emb = self.noise_emb(t_batch * 1.0)
            noise_emb = self.noise_emb(t)
            # t = t.view(-1, 1) 
            self.check("obs_before_cat", obs)
            self.check("x_t_before_cat", x_t)
            self.check("t_before_cat", t)
            x_in = torch.cat([obs, x_t], dim=-1)
            self.check("x_in", x_in)
            if self.check_params(self.actor_mlp):
                print("💥 WEIGHTS NAN BEFORE FORWARD")
            hidden = self.actor_mlp(x_in)
            self.check("actor_mlp", hidden)
            hidden = self.actor_norm(hidden, noise_emb)
            self.check("actor_norm", hidden)
            hidden = self.post_adaln_non_linearity(hidden)
            self.check("post_adaln", hidden)
            velocity = self.mu(hidden)
            self.check("mu", velocity)

            return velocity
        
        def eval_critic(self, obs):
            c_out = obs
            if(type(c_out) == dict): #ZC9
                c_out = c_out['obs']
            c_out = c_out.view(c_out.size(0), -1)
            c_out = self.critic_mlp(c_out)              
            # value = self.value_act(self.value(c_out))
            value = self.value(c_out)
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
            return x_t

    def build(self, name, **kwargs):
        net = FlowMatchingOptimBuilder.Network(self.params, **kwargs)
        return net
