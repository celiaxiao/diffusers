# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
import tqdm

from ...models import TemporalUnet
from ...pipelines import DiffusionPipeline
from ...utils import randn_tensor
from ...utils.dummy_pt_objects import DDPMScheduler


class ValueGuidedPipeline(DiffusionPipeline):
    r"""
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)
    Pipeline for sampling actions from a diffusion model trained to predict sequences of states.

    Original implementation inspired by this repository: https://github.com/jannerm/diffuser.

    Parameters:
        value_function ([`UNet1DModel`]): A specialized UNet for fine-tuning trajectories base on reward.
        unet ([`UNet1DModel`]): U-Net architecture to denoise the encoded trajectories.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded trajectories. Default for this
            application is [`DDPMScheduler`].
        env: An environment following the OpenAI gym API to act in. For now only Hopper has pretrained models.
    """

    def __init__(
        self,
        value_function: TemporalUnet,
        unet: TemporalUnet,
        scheduler: DDPMScheduler,
        env,
        normalizer,
        env_stat,
        horizon = 32,
        n_guide_steps = 2,
        scale=0.1,
    ):
        super().__init__()
        self.value_function = value_function
        self.unet = unet
        self.scheduler = scheduler
        self.env = env
        self.normalizer = normalizer
        self.planning_horizon = horizon
        self.n_guide_steps = n_guide_steps
        self.scale = scale
        # self.data = env.get_dataset()
        # self.means = dict()
        # for key in self.data.keys():
        #     try:
        #         self.means[key] = self.data[key].mean()
        #     except:  # noqa: E722
        #         pass
        # self.stds = dict()
        # for key in self.data.keys():
        #     try:
        #         self.stds[key] = self.data[key].std()
        #     except:  # noqa: E722
        #         pass
        self.state_dim = env_stat.observation_dim
        self.action_dim = env_stat.action_dim
        # TODO:
        self.state_dim = 16

    def normalize(self, x_in, key):
        return self.normalizer.normalize(x_in, key=key)
        # return (x_in - self.means[key]) / self.stds[key]

    def de_normalize(self, x_in, key):
        return self.normalizer.unnormalize(x_in, key=key)
        # return x_in * self.stds[key] + self.means[key]

    def to_torch(self, x_in):
        if type(x_in) is dict:
            return {k: self.to_torch(v) for k, v in x_in.items()}
        elif torch.is_tensor(x_in):
            return x_in.to(self.unet.device)
        return torch.tensor(x_in, device=self.unet.device)

    def reset_x0(self, x_in, cond, act_dim):
        for key, val in cond.items():
            x_in[:, key, act_dim:] = val.clone()
        return x_in

    def run_diffusion(self, x, conditions, n_guide_steps, scale):
        batch_size = x.shape[0]
        y = None
        for i in tqdm.tqdm(self.scheduler.timesteps):
            # create batch of timesteps to pass into model
            timesteps = torch.full((batch_size,), i, device=self.unet.device, dtype=torch.long)
            for _ in range(n_guide_steps):
                with torch.enable_grad():
                    x.requires_grad_()

                    y = self.value_function(x, conditions, timesteps)
                    grad = torch.autograd.grad([y.sum()], [x])[0]

                    posterior_variance = self.scheduler._get_variance(i)
                    model_std = torch.exp(0.5 * posterior_variance)
                    grad = model_std * grad

                grad[timesteps < 2] = 0
                grad_token = grad.reshape(x.shape)
                x = x.detach()
                x = x + scale * grad_token
                x = self.reset_x0(x, conditions, self.action_dim)
            with torch.no_grad():
                prev_x = self.unet(x, conditions, timesteps)

            # TODO: verify deprecation of this kwarg
            x = self.scheduler.step(prev_x, i, x,
            #  predict_epsilon=False
             )["prev_sample"]

            # apply conditions to the trajectory (set the initial state)
            x = self.reset_x0(x, conditions, self.action_dim)
            x = self.to_torch(x)
        return x, y

    def __call__(self, obs, batch_size=64):
        # normalize the observations and create  batch dimension
        obs = self.normalize(obs, "observations")
        # TODO:
        obs = obs[None].repeat(batch_size, axis=0)
        conditions = {0: self.to_torch(obs)}
        shape = (batch_size, self.planning_horizon, self.state_dim + self.action_dim)

        # generate initial noise and apply our conditions (to make the trajectories start at current state)
        x1 = randn_tensor(shape, device=self.unet.device)
        # print(f"X1: {x1.shape}, {conditions[0].shape=}")
        x = self.reset_x0(x1, conditions, self.action_dim)
        x = self.to_torch(x)

        # run the diffusion process
        x, y = self.run_diffusion(x, conditions, self.n_guide_steps, self.scale)
        # sort output trajectories by value
        if y is not None:
            sorted_idx = y.argsort(0, descending=True).squeeze()
            sorted_values = x[sorted_idx]
        else: # use the first output if we're not using guided steps
            sorted_values = x
        actions = sorted_values[:, :, : self.action_dim]
        actions = actions.detach().cpu().numpy()
        denorm_actions = self.de_normalize(actions, key="actions")

        # select the action with the highest value
        if y is not None:
            selected_index = 0
        else:
            # if we didn't run value guiding, select a random action
            selected_index = np.random.randint(0, batch_size)
        denorm_actions = denorm_actions[selected_index, 0]
        return denorm_actions, sorted_values

class ValueGuidedSlotsPipeline(ValueGuidedPipeline):
    def __init__(
        self,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

    def reset_x0(self, x_in, cond, act_dim):
        for key, val in cond.items():
            x_in[:, key, :, act_dim:] = val.clone()
        return x_in

    def __call__(self, obs, batch_size=64):
        '''
        obs: (batch_size, num_slots, state_dim)
        '''
        # normalize the observations and create  batch dimension
        obs = self.normalize(obs, "observations")
        # TODO: used for token pipeline only
        num_objects = obs.shape[0]
        obs = obs[None].repeat(batch_size, axis=0)
        conditions = {0: self.to_torch(obs)}
        shape = (batch_size, self.planning_horizon, num_objects, self.state_dim + self.action_dim)

        # generate initial noise and apply our conditions (to make the trajectories start at current state)
        x1 = randn_tensor(shape, device=self.unet.device)
        x = self.reset_x0(x1, conditions, self.action_dim)
        x = self.to_torch(x)

        # run the diffusion process [B, H, N, state_dim + action_dim]
        x, y = self.run_diffusion(x, conditions, self.n_guide_steps, self.scale)
        # sort output trajectories by value
        if y is not None:
            sorted_idx = y.argsort(0, descending=True).squeeze()
            sorted_values = x[sorted_idx]
        else: # use the first output if we're not using guided steps
            sorted_values = x
        # [B, H, N, action_dim + state_dim] -> [B, H, N, action_dim ]
        actions = sorted_values[:, :, :, : self.action_dim]
        actions = actions.detach().cpu().numpy()
        denorm_actions = self.de_normalize(actions, key="actions")

        # select the action with the highest value
        if y is not None:
            selected_index = 0
        else:
            # if we didn't run value guiding, select a random action
            selected_index = np.random.randint(0, batch_size)
        denorm_actions = denorm_actions[selected_index, 0]
        return denorm_actions, sorted_values

class ValueGuidedHistoryPipeline(ValueGuidedPipeline):
    def __init__(
        self, history_len=1,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.history_len = history_len
    
    def get_conditions(self, observations, actions):
        '''
            condition on current observation for planning
        '''
        # condition on the history_len observations
        condition = {}
        for i in range(self.history_len):
            condition[i] = np.concatenate([actions[i], observations[i]], axis=-1)
        return condition

    def reset_x0(self, x_in, cond, act_dim):
        for key, val in cond.items():
            x_in[:, key] = val.clone()
        return x_in

    def __call__(self, obs, batch_size=64):
        '''
        obs: (batch_size, num_slots, state_dim)
        '''
        # normalize the observations and create  batch dimension
        obs = self.normalize(obs, "observations")
        # TODO: used for token pipeline only
        num_objects = obs.shape[0]
        obs = obs[None].repeat(batch_size, axis=0)
        conditions = self.get_conditions(obs)
        shape = (batch_size, self.planning_horizon, num_objects, self.state_dim + self.action_dim)

        # generate initial noise and apply our conditions (to make the trajectories start at current state)
        x1 = randn_tensor(shape, device=self.unet.device)
        x = self.reset_x0(x1, conditions, self.action_dim)
        x = self.to_torch(x)

        # run the diffusion process [B, H, N, state_dim + action_dim]
        x, y = self.run_diffusion(x, conditions, self.n_guide_steps, self.scale)
        # sort output trajectories by value
        if y is not None:
            sorted_idx = y.argsort(0, descending=True).squeeze()
            sorted_values = x[sorted_idx]
        else: # use the first output if we're not using guided steps
            sorted_values = x
        # [B, H, N, action_dim + state_dim] -> [B, H, N, action_dim ]
        actions = sorted_values[:, :, :, : self.action_dim]
        actions = actions.detach().cpu().numpy()
        denorm_actions = self.de_normalize(actions, key="actions")

        # select the action with the highest value
        if y is not None:
            selected_index = 0
        else:
            # if we didn't run value guiding, select a random action
            selected_index = np.random.randint(0, batch_size)
        denorm_actions = denorm_actions[selected_index, 0]
        return denorm_actions, sorted_values
