import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces

from .base_class import BaseAlgorithm
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.vec_env import VecEnv

SelfOnPolicyAlgorithm = TypeVar("SelfOnPolicyAlgorithm", bound="OnPolicyAlgorithm")


class OnPolicyAlgorithm(BaseAlgorithm):
    """
    The base for On-Policy algorithms (ex: A2C/PPO).

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator.
        Equivalent to classic advantage when set to 1.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param monitor_wrapper: When creating an environment, whether to wrap it
        or not in a Monitor wrapper.
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    :param supported_action_spaces: The action spaces supported by the algorithm.
    """

    rollout_buffer: RolloutBuffer
    policy: ActorCriticPolicy

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule],
        n_steps: int,
        gamma: float,
        gae_lambda: float,
        ent_coef: float,
        vf_coef: float,
        max_grad_norm: float,
        use_sde: bool,
        sde_sample_freq: int,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        monitor_wrapper: bool = True,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        supported_action_spaces: Optional[Tuple[Type[spaces.Space], ...]] = None,
    ):
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            support_multi_env=True,
            seed=seed,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            supported_action_spaces=supported_action_spaces
        )

        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.reward_record = {1: [], 2: []}

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        buffer_cls = DictRolloutBuffer if isinstance(self.observation_space, spaces.Dict) else RolloutBuffer

        self.rollout_buffer_p1 = buffer_cls(
            self.n_steps,
            self.observation_space,  # type: ignore[arg-type]
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )
        self.rollout_buffer_p2 = buffer_cls(
            self.n_steps,
            self.observation_space,  # type: ignore[arg-type]
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )
        self.policy_p1 = self.policy_class(  # type: ignore[assignment]
            self.observation_space, self.action_space, self.lr_schedule, use_sde=self.use_sde, **self.policy_kwargs
        )
        self.policy_p1 = self.policy_p1.to(self.device)
        self.policy_p2 = self.policy_class(  # type: ignore[assignment]
            self.observation_space, self.action_space, self.lr_schedule, use_sde=self.use_sde, **self.policy_kwargs
        )
        self.policy_p2 = self.policy_p2.to(self.device)

    def forward_p1(
        self,
        env: VecEnv,
        n_steps,
        deterministic=False
    ) -> bool:
        if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
            # Sample a new noise matrix
            self.policy_p1.reset_noise(env.num_envs)

        with th.no_grad():
            # Convert to pytorch tensor or to TensorDict
            obs_tensor = obs_as_tensor(self._last_obs, self.device)
            actions_p1, values_p1, log_probs_p1 = self.policy_p1(obs_tensor, deterministic=deterministic)
        actions_p1 = actions_p1.cpu().numpy()

        # Rescale and perform action
        clipped_actions_p1 = actions_p1

        if isinstance(self.action_space, spaces.Box):
            if self.policy_p1.squash_output:
                # Unscale the actions to match env bounds
                # if they were previously squashed (scaled in [-1, 1])
                clipped_actions_p1 = self.policy_p1.unscale_action(clipped_actions_p1)
            else:
                # Otherwise, clip the actions to avoid out of bound error
                # as we are sampling from an unbounded Gaussian distribution
                clipped_actions_p1 = np.clip(actions_p1, self.action_space.low, self.action_space.high)
        return actions_p1, values_p1, log_probs_p1, clipped_actions_p1
    
    def forward_p2(
        self,
        env: VecEnv,
        n_steps,
        deterministic=False
    ) -> bool:
        if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
            # Sample a new noise matrix
            self.policy_p2.reset_noise(env.num_envs)

        with th.no_grad():
            # Convert to pytorch tensor or to TensorDict
            obs_tensor = obs_as_tensor(self._last_obs, self.device)
            actions_p2, values_p2, log_probs_p2 = self.policy_p2(obs_tensor, deterministic=deterministic)
        actions_p2 = actions_p2.cpu().numpy()

        # Rescale and perform action
        clipped_actions_p2 = actions_p2

        if isinstance(self.action_space, spaces.Box):
            if self.policy_p2.squash_output:
                # Unscale the actions to match env bounds
                # if they were previously squashed (scaled in [-1, 1])
                clipped_actions_p2 = self.policy_p2.unscale_action(clipped_actions_p2)
            else:
                # Otherwise, clip the actions to avoid out of bound error
                # as we are sampling from an unbounded Gaussian distribution
                clipped_actions_p2 = np.clip(actions_p2, self.action_space.low, self.action_space.high)
        return actions_p2, values_p2, log_probs_p2, clipped_actions_p2

    def update_p1(self, infos, dones, actions_p1,
                    rewards_p1, rollout_buffer_p1, 
                    values_p1, log_probs_p1):
        if isinstance(self.action_space, spaces.Discrete):
            actions_p1 = actions_p1.reshape(-1, 1)
        
        # for idx, done in enumerate(dones):
        #     if (
        #         done
        #         and infos[idx].get("terminal_observation") is not None
        #         and infos[idx].get("TimeLimit.truncated", False)
        #     ):
        #         terminal_obs = self.policy_p1.obs_to_tensor(infos[idx]["terminal_observation"])[0]
        #         with th.no_grad():
        #             terminal_p1_value = self.policy_p1.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
        #         rewards_p1[idx] += self.gamma * terminal_p1_value

        rollout_buffer_p1.add(
            self._last_obs,  # type: ignore[arg-type]
            actions_p1,
            rewards_p1,
            self._last_episode_starts,  # type: ignore[arg-type]
            values_p1,
            log_probs_p1,
        )
        return rollout_buffer_p1, rewards_p1, actions_p1

    def update_p2(self, infos, dones, actions_p2,
                    rewards_p2, rollout_buffer_p2, 
                    values_p2, log_probs_p2):
        if isinstance(self.action_space, spaces.Discrete):
            # Reshape in case of discrete action
            actions_p2 = actions_p2.reshape(-1, 1)
        # Handle timeout by bootstraping with value function
        # see GitHub issue #633
        # for idx, done in enumerate(dones):
        #     if (
        #         done
        #         and infos[idx].get("terminal_observation") is not None
        #         and infos[idx].get("TimeLimit.truncated", False)
        #     ):
        #         terminal_obs = self.policy_p2.obs_to_tensor(infos[idx]["terminal_observation"])[0]
        #         with th.no_grad():
        #             terminal_p2_value = self.policy_p2.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
        #         rewards_p2[idx] += self.gamma * terminal_p2_value

        rollout_buffer_p2.add(
            self._last_obs,  # type: ignore[arg-type]
            actions_p2,
            rewards_p2,
            self._last_episode_starts,  # type: ignore[arg-type]
            values_p2,
            log_probs_p2,
        )
        return rollout_buffer_p2, rewards_p2, actions_p2
        

    def train(self) -> None:
        """
        Consume current rollout data and update policy parameters.
        Implemented by individual algorithms.
        """
        raise NotImplementedError

    def learn(
        self: SelfOnPolicyAlgorithm,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "OnPolicyAlgorithm",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfOnPolicyAlgorithm:
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        assert self.env is not None

        infos = None
        p1_n_steps, p2_n_steps = 0, 0
        env, n_rollout_steps = self.env, self.n_steps
        action_memory_p1 = []
        action_memory_p2 = []
        diversity_p1 = []
        diversity_p2 = []
        win = []
        win_rate = 0
        training_player = 1
        acting_p1 = False
        acting_p2 = False
        self.defense_reward_p1_list = []
        self.defense_reward_p2_list = []
        self.logger.record("training_player", training_player)

        while self.num_timesteps < total_timesteps:
            # 获取当前状态下预测的action与value
            if infos is None:
                rewards_p1_cum, rewards_p2_cum = 0, 0
                actions_p1, values_p1, log_probs_p1, clipped_actions_p1 = self.forward_p1(env, p1_n_steps, deterministic=False)
                actions_p2, values_p2, log_probs_p2, clipped_actions_p2 = self.forward_p2(env, p2_n_steps, deterministic=False)
                self.reward_record['action_p1'] = clipped_actions_p1
                self.reward_record['action_p2'] = clipped_actions_p2
                action_memory_p1.append(clipped_actions_p1[0])
                action_memory_p2.append(clipped_actions_p2[0])
            else:
                if p1_next_step and not acting_p1:
                    actions_p1, values_p1, log_probs_p1, clipped_actions_p1 = self.forward_p1(env, p1_n_steps, deterministic=False)
                    action_memory_p1.append(clipped_actions_p1[0])
                    if len(action_memory_p1) > 10:
                        action_memory_p1.pop(0)
                if p2_next_step and not acting_p2:
                    actions_p2, values_p2, log_probs_p2, clipped_actions_p2 = self.forward_p2(env, p2_n_steps, deterministic=False)
                    action_memory_p2.append(clipped_actions_p2[0])
                    if len(action_memory_p2) > 10:
                        action_memory_p2.pop(0)
            # 将action与环境交互
            new_obs, (rewards_p1, rewards_p2), dones, infos = env.step([np.concatenate([clipped_actions_p1, clipped_actions_p2])])
            clipped_actions_p1, clipped_actions_p2 = np.array([-1]), np.array([-1])
            self.reward_cum(infos[0])
            # combo是否结束
            p1_next_step, p2_next_step = infos[0]['p1_next_step'], infos[0]['p2_next_step']
            # 累积combo内的reward
            rewards_p1_cum = rewards_p1_cum + rewards_p1
            rewards_p2_cum = rewards_p2_cum + rewards_p2
            if infos[0]['win'] == 'TRUE':
                win.append(1)
            elif infos[0]['win'] == 'FALSE':
                win.append(2)
            if len(win) > 100:
                win.pop(0)
            
            if infos[0]['win'] != 'UNKNOWN':
                win_rate = np.mean(np.array(win[-50:]) == training_player)
                if len(win) >= 50 and win_rate > 0.8:
                    training_player = 3-training_player
                    if training_player == 1:
                        self.policy_p1 = self.policy_class(  # type: ignore[assignment]
                            self.observation_space, self.action_space, self.lr_schedule, use_sde=self.use_sde, **self.policy_kwargs
                        )
                        self.policy_p1 = self.policy_p1.to(self.device)
                    else:
                        self.policy_p2 = self.policy_class(  # type: ignore[assignment]
                            self.observation_space, self.action_space, self.lr_schedule, use_sde=self.use_sde, **self.policy_kwargs
                        )
                        self.policy_p2 = self.policy_p2.to(self.device)
                    print("[STEP %08d]: player %d win rate %02.2f, start train player %d"%(self.num_timesteps, 3-training_player, win_rate*100, training_player))
                    self.logger.record("training_player", training_player)
            
            if p1_next_step:
                # print("[STEP %08d]: training player %d, win rate %02.2f"%(self.num_timesteps, training_player, win_rate*100))
                rewards_p1_cum = self.reward_cum_end(player=1)
                diversity_p1.append(len(set(action_memory_p1)))
                # rewards_p1_cum = rewards_p1_cum + 0.01*(len(set(action_memory_p1))-5)
                self.rollout_buffer_p1, rewards_p1, actions_p1 = self.update_p1(infos, dones, actions_p1,
                        rewards_p1_cum, self.rollout_buffer_p1, 
                        values_p1, log_probs_p1)
                rewards_p1_cum = 0
                p1_n_steps = p1_n_steps + 1
                self.num_timesteps += env.num_envs
                callback.update_locals(locals())
                if callback.on_step() is False:
                    return False
                self._update_info_buffer(infos)
                if p1_n_steps == n_rollout_steps:
                    p1_n_steps = 0
                    with th.no_grad():
                        values_p1 = self.policy_p1.predict_values(obs_as_tensor(new_obs, self.device), 
                                                            th.tensor(actions_p1[:, 0]).to(self.device))  # type: ignore[arg-type]

                    self.rollout_buffer_p1.compute_returns_and_advantage(last_values=values_p1, dones=dones)
                    callback.update_locals(locals())
                    callback.on_rollout_end()
                    self._update_current_progress_remaining(self.num_timesteps, total_timesteps)
                    if log_interval is not None and iteration % log_interval == 0:
                        assert self.ep_p1_info_buffer is not None
                        if len(self.ep_p1_info_buffer) > 0 and len(self.ep_p1_info_buffer[0]) > 0:
                            self.logger.record("rollout/ep_p1_rew_mean", safe_mean([ep_info["r_p1"] for ep_info in self.ep_p1_info_buffer]))
                            self.logger.record("rollout/ep_p1_def_mean", safe_mean(self.defense_reward_p1_list))
                            self.logger.record("rollout/ep_p1_div_mean", safe_mean(diversity_p1))
                            self.logger.record("rollout/ep_p1_win_mean", safe_mean(np.array(win[-50:])==1))
                            diversity_p1 = []
                            self.defense_reward_p1_list = []
                        self.logger.dump(step=self.num_timesteps)
                    if training_player==1:
                        self.train_p1()
                    self.rollout_buffer_p1.reset()
            
            if p2_next_step:
                # print("[STEP %08d]: training player %d, win rate %02.2f"%(self.num_timesteps, training_player, win_rate*100))
                rewards_p2_cum = self.reward_cum_end(player=2)
                diversity_p2.append(len(set(action_memory_p2)))
                # rewards_p2_cum = rewards_p2_cum + 0.01*(len(set(action_memory_p2))-5)
                self.rollout_buffer_p2, rewards_p2, actions_p2 = self.update_p2(infos, dones, actions_p2,
                        rewards_p2_cum, self.rollout_buffer_p2, 
                        values_p2, log_probs_p2)
                rewards_p2_cum = 0
                p2_n_steps = p2_n_steps + 1
                self.num_timesteps += env.num_envs
                callback.update_locals(locals())
                if callback.on_step() is False:
                    return False
                self._update_info_buffer(infos)
                if p2_n_steps == n_rollout_steps:
                    p2_n_steps = 0
                    with th.no_grad():
                        # Compute value for the last timestep
                        values_p2 = self.policy_p2.predict_values(obs_as_tensor(new_obs, self.device), 
                                                            th.tensor(actions_p2[:, 0]).to(self.device))  # type: ignore[arg-type]

                    self.rollout_buffer_p2.compute_returns_and_advantage(last_values=values_p2, dones=dones)
                    callback.update_locals(locals())
                    callback.on_rollout_end()
                    self._update_current_progress_remaining(self.num_timesteps, total_timesteps)
                    if log_interval is not None and iteration % log_interval == 0:
                        assert self.ep_p2_info_buffer is not None
                        if len(self.ep_p2_info_buffer) > 0 and len(self.ep_p2_info_buffer[0]) > 0:
                            self.logger.record("rollout/ep_p2_rew_mean", safe_mean([ep_info["r_p2"] for ep_info in self.ep_p2_info_buffer]))
                            self.logger.record("rollout/ep_p2_def_mean", safe_mean(self.defense_reward_p2_list))
                            self.logger.record("rollout/ep_p2_div_mean", safe_mean(diversity_p2))
                            self.logger.record("rollout/ep_p2_win_mean", safe_mean(np.array(win[-50:])==2))
                            diversity_p2 = []
                            self.defense_reward_p2_list = []
                        self.logger.dump(step=self.num_timesteps)
                    if training_player==2:
                        self.train_p2()
                    self.rollout_buffer_p2.reset()
            
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        callback.on_training_end()

        return self

    def reward_cum(self, info):
        damage_reward_p1 = info['damage_reward_p1']
        defense_reward_p1 = max(0, info['p1_defense']*2)
        reward_p1 = damage_reward_p1 + defense_reward_p1
        self.reward_record[1].append(reward_p1)
        self.defense_reward_p1_list.append(defense_reward_p1)
        
        damage_reward_p2 = info['damage_reward_p2']
        defense_reward_p2 = max(0, info['p2_defense']*2)
        reward_p2 = damage_reward_p2 + defense_reward_p2
        self.defense_reward_p2_list.append(defense_reward_p2)
        self.reward_record[2].append(reward_p2)

    def reward_cum_end(self, player):
        reward = sum(self.reward_record[player])
        self.reward_record[player] = []
        return reward


    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy_p1", "policy_p1.optimizer", 
                       "policy_p2", "policy_p2.optimizer"]

        return state_dicts, []
