import warnings
from typing import Any, ClassVar, Dict, Optional, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn
from .cnn_policy import ActorCriticCnnPolicy
from .on_policy_algorithm import OnPolicyAlgorithm

SelfPPO = TypeVar("SelfPPO", bound="PPO")


class PPO(OnPolicyAlgorithm):
    """
    Proximal Policy Optimization algorithm (PPO) (clip version)

    Paper: https://arxiv.org/abs/1707.06347
    Code: This implementation borrows code from OpenAI Spinning Up (https://github.com/openai/spinningup/)
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
    Stable Baselines (PPO2 from https://github.com/hill-a/stable-baselines)

    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
        NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization)
        See https://github.com/pytorch/pytorch/issues/29372
    :param batch_size: Minibatch size
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param clip_range: Clipping parameter, it can be a function of the current progress
        remaining (from 1 to 0).
    :param clip_range_vf: Clipping parameter for the value function,
        it can be a function of the current progress remaining (from 1 to 0).
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        no clipping will be done on the value function.
        IMPORTANT: this clipping depends on the reward scaling.
    :param normalize_advantage: Whether to normalize or not the advantage
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param target_kl: Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        # Sanity check, otherwise it will lead to noisy gradient and NaN
        # because of the advantage normalization
        if normalize_advantage:
            assert (
                batch_size > 1
            ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"

        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.env.num_envs * self.n_steps
            assert buffer_size > 1 or (
                not normalize_advantage
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

        
    def train_p1(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy_p1.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy_p1.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_p1_losses = []
        pg_p1_losses, value_p1_losses = [], []
        clip_fractions_p1 = []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs_p1 = []
            # Do a complete pass on the rollout buffer
            for rollout_data_p1 in self.rollout_buffer_p1.get(self.batch_size):
                actions_p1 = rollout_data_p1.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions_p1 = rollout_data_p1.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy_p1.reset_noise(self.batch_size)

                values_p1, log_prob_p1, entropy_p1 = self.policy_p1.evaluate_actions(rollout_data_p1.observations, actions_p1)
                values_p1 = values_p1.flatten()
                advantages_p1 = rollout_data_p1.advantages
                
                # 计算策略损失
                if self.normalize_advantage and len(advantages_p1) > 1:
                    advantages_p1 = (advantages_p1 - advantages_p1.mean()) / (advantages_p1.std() + 1e-8)
                ratio_p1 = th.exp(log_prob_p1 - rollout_data_p1.old_log_prob)
                policy_p1_loss_1 = advantages_p1 * ratio_p1
                policy_p1_loss_2 = advantages_p1 * th.clamp(ratio_p1, 1 - clip_range, 1 + clip_range)
                policy_p1_loss = -th.min(policy_p1_loss_1, policy_p1_loss_2).mean()

                # Logging
                pg_p1_losses.append(policy_p1_loss.item())
                clip_fraction_p1 = th.mean((th.abs(ratio_p1 - 1) > clip_range).float()).item()
                clip_fractions_p1.append(clip_fraction_p1)

                # 时序差分算法计算优化value
                if self.clip_range_vf is None:
                    values_pred_p1 = values_p1
                else:
                    values_pred_p1 = rollout_data_p1.old_values + th.clamp(
                        values_p1 - rollout_data_p1.old_values, -clip_range_vf, clip_range_vf
                    )
                value_p1_loss = F.mse_loss(rollout_data_p1.returns, values_pred_p1)
                value_p1_losses.append(value_p1_loss.item())

                # 熵损失让决策更集中
                if entropy_p1 is None:
                    entropy_p1_loss = -th.mean(-log_prob_p1)
                else:
                    entropy_p1_loss = -th.mean(entropy_p1)

                entropy_p1_losses.append(entropy_p1_loss.item())

                loss_p1 = policy_p1_loss + self.ent_coef * entropy_p1_loss + self.vf_coef * value_p1_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio_p1 = log_prob_p1 - rollout_data_p1.old_log_prob
                    approx_kl_div_p1 = th.mean((th.exp(log_ratio_p1) - 1) - log_ratio_p1).cpu().numpy()
                    approx_kl_divs_p1.append(approx_kl_div_p1)

                if self.target_kl is not None and approx_kl_div_p1 > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div_p1:.2f}")
                    break

                # Optimization step
                self.policy_p1.optimizer.zero_grad()
                loss_p1.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy_p1.parameters(), self.max_grad_norm)
                self.policy_p1.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break
            
            
    def train_p2(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy_p2.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy_p1.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_p2_losses = []
        pg_p2_losses, value_p2_losses = [], []
        clip_fractions_p2 = []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs_p2 = []
            # Do a complete pass on the rollout buffer
            for rollout_data_p2 in self.rollout_buffer_p2.get(self.batch_size):
                actions_p2 = rollout_data_p2.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions_p2 = rollout_data_p2.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy_p2.reset_noise(self.batch_size)

                values_p2, log_prob_p2, entropy_p2 = self.policy_p2.evaluate_actions(rollout_data_p2.observations, actions_p2)
                values_p2 = values_p2.flatten()
                # Normalize advantage
                advantages_p2 = rollout_data_p2.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages_p2) > 1:
                    advantages_p2 = (advantages_p2 - advantages_p2.mean()) / (advantages_p2.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio_p2 = th.exp(log_prob_p2 - rollout_data_p2.old_log_prob)

                # clipped surrogate loss
                policy_p2_loss_1 = advantages_p2 * ratio_p2
                policy_p2_loss_2 = advantages_p2 * th.clamp(ratio_p2, 1 - clip_range, 1 + clip_range)
                policy_p2_loss = -th.min(policy_p2_loss_1, policy_p2_loss_2).mean()

                # Logging
                pg_p2_losses.append(policy_p2_loss.item())
                clip_fraction_p2 = th.mean((th.abs(ratio_p2 - 1) > clip_range).float()).item()
                clip_fractions_p2.append(clip_fraction_p2)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred_p2 = values_p2
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred_p2 = rollout_data_p2.old_values + th.clamp(
                        values_p2 - rollout_data_p2.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_p2_loss = F.mse_loss(rollout_data_p2.returns, values_pred_p2)
                value_p2_losses.append(value_p2_loss.item())

                # Entropy loss favor exploration
                if entropy_p2 is None:
                    # Approximate entropy when no analytical form
                    entropy_p2_loss = -th.mean(-log_prob_p2)
                else:
                    entropy_p2_loss = -th.mean(entropy_p2)

                entropy_p2_losses.append(entropy_p2_loss.item())

                loss_p2 = policy_p2_loss + self.ent_coef * entropy_p2_loss + self.vf_coef * value_p2_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio_p2 = log_prob_p2 - rollout_data_p2.old_log_prob
                    approx_kl_div_p2 = th.mean((th.exp(log_ratio_p2) - 1) - log_ratio_p2).cpu().numpy()
                    approx_kl_divs_p2.append(approx_kl_div_p2)

                if self.target_kl is not None and approx_kl_div_p2 > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div_p2:.2f}")
                    break

                # Optimization step
                self.policy_p2.optimizer.zero_grad()
                loss_p2.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy_p2.parameters(), self.max_grad_norm)
                self.policy_p2.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

    def learn(
        self: SelfPPO,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "PPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfPPO:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
