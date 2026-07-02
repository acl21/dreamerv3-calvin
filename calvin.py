import copy
from collections import defaultdict
from contextlib import contextmanager

import hydra
import numpy as np
from gym import spaces
from calvin_env.calvin_env.envs.play_table_env import PlayTableSimEnv

# import torch
import cv2

LETTERS_TO_SKILLS = {
    "A": "open_drawer",
    "B": "turn_on_lightbulb",
    "C": "move_slider_left",
    "D": "turn_on_led",
    "E": "close_drawer",
    "F": "turn_off_lightbulb",
    "G": "move_slider_right",
    "H": "turn_off_led",
}


class CalvinEnv(PlayTableSimEnv):
    def __init__(self, tasks: dict = {}, task_order: str = "", **kwargs):
        self.max_episode_steps = kwargs.pop("max_episode_steps")
        self.reward_norm = kwargs.pop("reward_norm")
        super().__init__(**kwargs)

        self.gripper_width = 64
        self.action_space = spaces.Box(low=-1, high=1, shape=(7,))
        self.observation_space = spaces.Box(low=-1, high=1, shape=(21,))
        # self.observation_space = spaces.Box(low=-1, high=1, shape=(533,))
        # self.observation_space = spaces.Box(
        #     low=-1, high=1, shape=(self.gripper_width, self.gripper_width, 3)
        # )

        self.tasks = hydra.utils.instantiate(tasks)
        self.target_tasks = list(self.tasks.tasks.keys())
        task_order = [*task_order]
        self.target_tasks = [LETTERS_TO_SKILLS[skill] for skill in task_order]
        self.tasks_to_complete = copy.deepcopy(self.target_tasks)
        self.completed_tasks = []
        self.solved_subtasks = defaultdict(lambda: 0)
        self._t = 0
        self.sequential = True

        self.ee_noise = np.array([0.05, 0.05, 0.05])  # Units: meters
        self.init_pos = None
        # self.pt_encoder = PTEncoder("cifar10-resnet18").to("cuda")
        self.centroid = np.array([0.036, -0.13, 0.509])
        self.obs_type = "vector"
        self.frames = []

    def reset(self):
        obs = super().reset()
        self.calibrate_EE_start_state(self.get_state_obs()["robot_obs"][:3])
        self.robot.update_target_variables()
        self.start_info = self.get_info()
        self._t = 0
        self.tasks_to_complete = copy.deepcopy(self.target_tasks)
        self.completed_tasks = []
        self.solved_subtasks = defaultdict(lambda: 0)
        if len(self.frames) > 0:
            self.save_recording()
        self.frames = []
        return self.get_obs()

    def reset_to_state(self, state):
        return super().reset(robot_obs=state[:15], scene_obs=state[15:])

    def get_obs(self):
        return self.get_default_obs()

    def get_default_obs(self):
        obs = self.get_state_obs()
        return np.concatenate([obs["robot_obs"], obs["scene_obs"]])[:21]

    def get_custom_obs2(self):
        obs = super().get_obs()

        return obs["rgb_obs"]["rgb_static"]
        # return cv2.resize(
        #     obs["rgb_obs"]["rgb_static"],
        #     (self.gripper_width, self.gripper_width),
        #     interpolation=cv2.INTER_AREA,
        # )

    # def get_custom_obs(self):
    #     obs = super().get_obs()

    #     gripper_img = np.moveaxis(obs["rgb_obs"]["rgb_gripper"], 2, 0)
    #     gripper_img = torch.tensor(gripper_img).to("cuda")
    #     gripper_img = gripper_img.unsqueeze(0)
    #     with torch.no_grad():
    #         gripper_state = self.pt_encoder(gripper_img)
    #     return np.concatenate(
    #         [self.get_default_obs(), gripper_state.cpu().numpy().flatten()]
    #     )

    def _reward(self):
        current_info = self.get_info()
        completed_tasks = self.tasks.get_task_info_for_set(
            self.start_info, current_info, self.target_tasks
        )
        next_task = self.tasks_to_complete[0]

        reward = 0
        for task in list(completed_tasks):
            if self.sequential:
                if task == next_task:
                    reward += 1
                    self.tasks_to_complete.pop(0)
                    self.completed_tasks.append(task)
            else:
                if task in self.tasks_to_complete:
                    reward += 1
                    self.tasks_to_complete.remove(task)
                    self.completed_tasks.append(task)

        reward *= self.reward_norm
        r_info = {"reward": reward}
        return reward, r_info

    def _termination(self):
        """Indicates if the robot has completed all tasks. Should be called after _reward()."""
        done = len(self.tasks_to_complete) == 0
        d_info = {"success": done, "terminated": done}
        return done, d_info

    def _postprocess_info(self, info):
        """Sorts solved subtasks into separately logged elements."""
        for task in self.target_tasks:
            self.solved_subtasks[task] = (
                1 if task in self.completed_tasks or self.solved_subtasks[task] else 0
            )
        return info

    def step(self, action):
        """Performing a relative action in the environment
        input:
            action: 7 tuple containing
                    Position x, y, z.
                    Angle in rad x, y, z.
                    Gripper action
                    each value in range (-1, 1)
        output:
            observation, reward, done info
        """
        # Transform gripper action to discrete space
        env_action = action.copy()
        env_action[-1] = (int(action[-1] >= 0) * 2) - 1
        self.robot.apply_action(env_action)
        for _ in range(self.action_repeat):
            self.p.stepSimulation(physicsClientId=self.cid)
        self.scene.step()
        obs = self.get_obs()
        info = self.get_info()
        reward, r_info = self._reward()
        done, d_info = self._termination()
        info.update(r_info)
        info.update(d_info)
        self._t += 1
        if self._t >= self.max_episode_steps:
            done = True
        self.record_frame()
        return obs, reward, done, self._postprocess_info(info)
        # return self._obs(
        #     obs,
        #     np.float32(reward),
        #     self._postprocess_info(info),
        #     is_last=done,
        #     is_terminal=done,
        #     obs_type=self.obs_type,
        # )

    def _obs(
        self,
        obs,
        reward,
        info,
        is_first=False,
        is_last=False,
        is_terminal=False,
        obs_type="vector",
    ):
        if obs_type == "vector":
            return dict(
                vector=obs,
                reward=reward,
                is_first=is_first,
                is_last=is_last,
                is_terminal=is_terminal,
                log_reward=np.float32(info["reward"] if info else 0.0),
            )
        else:
            return dict(
                image=obs,
                reward=reward,
                is_first=is_first,
                is_last=is_last,
                is_terminal=is_terminal,
                log_reward=np.float32(info["reward"] if info else 0.0),
            )

    @contextmanager
    def val_mode(self):
        """Sets validation parameters if desired. To be used like: with env.val_mode(): ...<do something>..."""
        pass
        yield
        pass

    def get_episode_info(self):
        completed_tasks = (
            self.completed_tasks if len(self.completed_tasks) > 0 else [None]
        )
        info = dict(
            solved_subtask=completed_tasks, tasks_to_complete=self.tasks_to_complete
        )
        info.update(self.solved_subtasks)
        return info

    def sample_ee_pose(self):
        """Samples a random end effector pose within a small range around the initial pose."""
        offset = [0, 0, 0]
        np.random.seed(np.random.randint(0, 1000))
        offset[0] = np.random.uniform(-self.ee_noise[0], self.ee_noise[0], 1)[0]
        offset[1] = np.random.uniform(-self.ee_noise[1], self.ee_noise[1], 1)[0]
        offset[2] = np.random.uniform(0, self.ee_noise[2], 1)[0]
        gripper_pos = self.centroid + offset
        gripper_orn = self.robot.target_orn
        return gripper_pos, gripper_orn

    def custom_step(self, action):
        """Called only by calibrate_EE_start_state to perform a absolute steps in the environment."""
        # Transform gripper action to discrete space
        env_action = action.copy()
        env_action[-1] = (int(action[-1] >= 0) * 2) - 1
        self.robot.apply_action(env_action)
        for _ in range(self.action_repeat):
            self.p.stepSimulation(physicsClientId=self.cid)
        self.scene.step()
        return self.get_state_obs()["robot_obs"][:3]

    def calibrate_EE_start_state(self, obs, error_margin=0.01, max_checks=15):
        """Samples a random but good starting point and moves the end effector to that point."""
        ee_pos, ee_orn = self.sample_ee_pose()
        count = 0
        action = np.array([ee_pos, ee_orn, -1], dtype=object)
        while np.linalg.norm(obs[:3] - ee_pos) > error_margin:
            obs = self.custom_step(action)
            if count >= max_checks:
                print("CALVIN is struggling to place the EE at the right initial pose.")
                print("Current EE pos: ", obs[:3])
                print("Desired EE pos: ", ee_pos)
                # Sample and try again
                ee_pos, ee_orn = self.sample_ee_pose()
                action = np.array([ee_pos, ee_orn, -1], dtype=object)
                count = 0
            count += 1
        return obs

    def record_frame(self, obs_type="rgb", cam_type="static", size=200):
        """Record RGB obsservations"""
        rgb_obs, depth_obs = self.get_camera_obs()
        if obs_type == "rgb":
            frame = rgb_obs[f"{obs_type}_{cam_type}"]
        else:
            frame = depth_obs[f"{obs_type}_{cam_type}"]
        frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
        self.frames.append(frame)

    def save_recording(self):
        import os
        import imageio

        """Save recorded frames as a video"""
        if len(self.frames) == 0:
            # This shouldn't happen but if it does, the function
            # call exits gracefully
            return None
        fname = "policy_rollout.gif"
        kargs = {"duration": 33}
        fpath = os.path.join("/home/lagandua/projects/dreamerv3-calvin/", fname)
        imageio.mimsave(fpath, np.array(self.frames), "GIF", **kargs)
        return fpath
