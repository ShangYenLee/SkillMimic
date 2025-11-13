# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch
import torch.nn as nn
from rl_games.algos_torch.network_builder import NetworkBuilder
from rl_games.algos_torch.models import BaseModel, ModelA2CContinuousLogStd

class FPOModel(ModelA2CContinuousLogStd):
    def __init__(self, network):
        BaseModel.__init__(self)
        self.network_builder = network
 
    def build(self, config):
        net = self.network_builder.build('fpo', **config)
        for name, _ in net.named_parameters():
            print(name)
        return FPOModel.Network(net)

    class Network(ModelA2CContinuousLogStd.Network):
        def __init__(self, flow_network):
            super().__init__(flow_network)
            self.flow_network = flow_network
            # optional torch.compile for speed (PyTorch 2+)
            try:
                self.forward = torch.compile(mode="reduce-overhead")(self.forward)
            except Exception:
                pass
        
        def is_rnn(self):
            return getattr(self.flow_network, "is_rnn", lambda: False)()

        def get_value_layer(self):
            return getattr(self.flow_network, "get_value_layer", lambda: None)()

        def get_default_rnn_state(self):
            if hasattr(self.flow_network, "get_default_rnn_state"):
                return self.flow_network.get_default_rnn_state()
            return None
        
        def sample_action(self, *args, **kwargs):
            return self.flow_network.sample_action(*args, **kwargs)

            #return getattr(self.flow_network.Network, "sample_action", lambda: False)()
        
        def eval_critic(self, *args, **kwargs):
            return self.flow_network.eval_critic(*args, **kwargs)

        def forward(self, input_dict):
            """
            Called by FPOAgent.calc_gradients(batch_dict).
            Just delegate to flow_network.forward.
            """
            return self.flow_network(input_dict)
