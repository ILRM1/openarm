"""DirectRL environment for OpenArm bimanual reach task."""

import math
from dataclasses import MISSING

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_error_magnitude, quat_from_euler_xyz
from dextrah_lab.assets.openarm.openarm_bimanual import OPEN_ARM_CFG, OPEN_ARM_HIGH_PD_CFG

@configclass
class BimanualReachDirectEnvCfg(DirectRLEnvCfg):
    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1.0 / 120.0, render_interval=4)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.5)

    # robot (set by subclass or user)
    robot_cfg: ArticulationCfg = OPEN_ARM_HIGH_PD_CFG.replace(prim_path="/World/envs/env_.*/Robot").replace(
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                    "openarm_left_joint1": -0.24,
                    "openarm_left_joint2": -1.05,
                    "openarm_left_joint3": 0.74,
                    "openarm_left_joint4": 1.86,
                    "openarm_left_joint5": -1.12,
                    "openarm_left_joint6": 0.,
                    "openarm_left_joint7": 0.94,

                    "openarm_right_joint1": 0.24,
                    "openarm_right_joint2": 1.05,
                    "openarm_right_joint3": -0.74,
                    "openarm_right_joint4": 1.86,
                    "openarm_right_joint5": 1.12,
                    "openarm_right_joint6": 0.17,
                    "openarm_right_joint7": -0.94,

                    "openarm_left_finger_joint.*": 0.044,
                    "openarm_right_finger_joint.*": 0.044,
                },  # Close the gripper
        )
    )

    left_arm_joint_name = [
        "openarm_left_joint1", "openarm_left_joint2", "openarm_left_joint3",
        "openarm_left_joint4", "openarm_left_joint5", "openarm_left_joint6",
        "openarm_left_joint7",
    ]
    right_arm_joint_name = [
        "openarm_right_joint1", "openarm_right_joint2", "openarm_right_joint3",
        "openarm_right_joint4", "openarm_right_joint5", "openarm_right_joint6",
        "openarm_right_joint7", 
    ]

    # MDP
    decimation: int = 2
    episode_length_s: float = 24.0

    # spaces
    observation_space: int = 35   # 7 * 8
    action_space: int = 7        # 7 * 2
    state_space: int = 0

    # action scale (matches ManagerBased JointPositionAction scale=0.5)
    action_scale: float = 0.5

    # command resampling interval
    resampling_time_s: float = 4.0

    # command ranges — left arm (positive y side)
    left_pos_x: tuple = (0.0, 0.4)
    left_pos_y: tuple = (-0.1, 0.4)
    left_pos_z: tuple = (0.10, 0.50)
    left_roll:  tuple = (-math.pi / 2, math.pi / 2)
    left_pitch: tuple = (-math.pi / 2, math.pi / 2)
    left_yaw:   tuple = (-math.pi / 2, math.pi / 2)

    # command ranges — right arm (negative y side)
    right_pos_x: tuple = (0.15, 0.30)
    right_pos_y: tuple = (-0.25, -0.15)
    right_pos_z: tuple = (0.30, 0.50)
    right_roll:  tuple = (-math.pi / 2, math.pi / 2)
    right_pitch: tuple = (-math.pi / 2, math.pi / 2)
    right_yaw:   tuple = (-math.pi / 2, math.pi / 2)

    # reward weights (match ManagerBased defaults)
    w_left_pos:       float = -0.25
    w_right_pos:      float = -0.25
    w_left_pos_fine:  float =  0.20
    w_right_pos_fine: float =  0.20
    w_left_ori:       float = -0.25
    w_right_ori:      float = -0.25
    w_action_rate:    float = -0.0001
    w_joint_vel:      float = -0.0001
    pos_fine_std:     float =  0.10


class BimanualReachDirectEnv(DirectRLEnv):
    """Bimanual reach environment using DirectRL API.

    Obs (56D):
        left_joint_pos_rel (7) | right_joint_pos_rel (7)
        left_joint_vel     (7) | right_joint_vel     (7)
        left_ee_pose_cmd   (7) | right_ee_pose_cmd   (7)
        left_last_action   (7) | right_last_action   (7)

    Action (14D):
        left_arm_delta  (7) | right_arm_delta (7)
        applied as: target = default_joint_pos + action_scale * action
    """

    cfg: BimanualReachDirectEnvCfg

    def __init__(self, cfg: BimanualReachDirectEnvCfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # joint indices
        self._left_ids,  _ = self.robot.find_joints(["openarm_left_joint[1-7]"])
        self._right_ids, _ = self.robot.find_joints(["openarm_right_joint[1-7]"])
        self._arm_ids = self._left_ids + self._right_ids

        # EE body indices
        left_body,  _ = self.robot.find_bodies(["openarm_left_ee_tcp"])
        right_body, _ = self.robot.find_bodies(["openarm_right_ee_tcp"])
        self._left_ee_id  = left_body[0]
        self._right_ee_id = right_body[0]

        # buffers
        n = self.num_envs
        self._actions      = torch.zeros(n, self.cfg.action_space, device=self.device)
        self._prev_actions = torch.zeros(n, self.cfg.action_space, device=self.device)

        # EE target pose: (pos 3 + quat 4) = 7D, in env-local frame
        self._left_cmd  = torch.zeros(n, 7, device=self.device)
        self._right_cmd = torch.zeros(n, 7, device=self.device)
        self._left_cmd[:, 6]  = 1.0   # identity quaternion w=1
        self._right_cmd[:, 6] = 1.0

        # command timer (counts env steps until resample)
        self._cmd_timer = torch.zeros(n, device=self.device)
        self._resample_steps = int(self.cfg.resampling_time_s / (self.cfg.decimation * self.cfg.sim.dt))

        self.robot_start_joint_pos =torch.tensor([0.63, -0.35,  -0.24,  2.0, -0.54, 0.0, 1.1,
                                            -0.63, 0.35,  0.24,  2.0, 0.54, 0.0, -1.1], device=self.device)
        self.robot_start_joint_pos = self.robot_start_joint_pos.repeat(self.num_envs, 1).contiguous()
        self.robot_start_joint_vel = torch.zeros(self.robot_start_joint_pos.shape, device=self.device)

        self.actuated_dof_indices = list()
        for joint_name in (cfg.left_arm_joint_name+cfg.right_arm_joint_name):
            self.actuated_dof_indices.append(self.robot.joint_names.index(joint_name))

        joint_pos_limits = self.robot.root_physx_view.get_dof_limits().to(self.device)
        # NOTE: this arranges the limits to be in the same joint order as fabrics
        self.robot_dof_lower_limits = joint_pos_limits[..., 0][:, self.actuated_dof_indices]
        self.robot_dof_upper_limits = joint_pos_limits[..., 1][:, self.actuated_dof_indices]

    # ------------------------------------------------------------------
    # scene
    # ------------------------------------------------------------------
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self.robot

        # ground plane
        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/ground", ground_cfg)

        # dome light
        light_cfg = sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0)
        light_cfg.func("/World/light", light_cfg)

        # replicate environments and filter cross-env collisions
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=["/World/ground"])

    # ------------------------------------------------------------------
    # stepping
    # ------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone()

        self.left_target = self.robot.data.joint_pos[:, self._left_ids] + self.cfg.action_scale * self._actions[:, :7]
        #right_target = self.robot.data.joint_pos[:, self._right_ids] + self.cfg.action_scale * self._actions[:, 7:14]

        self.left_target = torch.clamp(self.left_target, min=self.robot_dof_lower_limits[:,:7], max=self.robot_dof_upper_limits[:,:7])

    def _apply_action(self):
        self.robot.set_joint_position_target(self.left_target,  joint_ids=self._left_ids)
        #self.robot.set_joint_position_target(right_target, joint_ids=self._right_ids)

    # ------------------------------------------------------------------
    # observations
    # ------------------------------------------------------------------
    def _get_observations(self) -> dict:
        # update command timer and resample if needed
        self._cmd_timer += 1
        resample_ids = (self._cmd_timer >= self._resample_steps).nonzero(as_tuple=False).squeeze(-1)
        if resample_ids.numel() > 0:
            self._resample_commands(resample_ids)
            self._cmd_timer[resample_ids] = 0

        #dpos  = self.robot.data.default_joint_pos
        left_pos_rel  = self.robot.data.joint_pos[:, self._left_ids] 
        #right_pos_rel = self.robot.data.joint_pos[:, self._right_ids]
        left_vel_rel  = self.robot.data.joint_vel[:, self._left_ids]
        #right_vel_rel = self.robot.data.joint_vel[:, self._right_ids]

        obs = torch.cat([
            left_pos_rel,
            #right_pos_rel,
            left_vel_rel,
            #right_vel_rel,
            self.robot.data.body_pos_w[:, self._left_ee_id, :]  - self.scene.env_origins,
            self.robot.data.body_quat_w[:, self._left_ee_id, :],
            self._left_cmd,
            #self._right_cmd,
            self._actions[:, :7],
            #self._actions[:, 7:],
        ], dim=-1)

        return obs

    # ------------------------------------------------------------------
    # rewards
    # ------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        # EE positions in world frame → subtract env origin for local frame
        left_ee_pos  = self.robot.data.body_pos_w[:, self._left_ee_id, :]  - self.scene.env_origins
        #right_ee_pos = self.robot.data.body_pos_w[:, self._right_ee_id, :] - self.scene.env_origins
        left_ee_quat  = self.robot.data.body_quat_w[:, self._left_ee_id, :]
        #right_ee_quat = self.robot.data.body_quat_w[:, self._right_ee_id, :]

        # position error
        left_pos_err  = torch.norm(left_ee_pos  - self._left_cmd[:, :3],  dim=-1)
        #right_pos_err = torch.norm(right_ee_pos - self._right_cmd[:, :3], dim=-1)

        # orientation error (angle in radians)
        left_ori_err  = quat_error_magnitude(left_ee_quat,  self._left_cmd[:, 3:])
        #right_ori_err = quat_error_magnitude(right_ee_quat, self._right_cmd[:, 3:])

        # action rate penalty
        action_rate = torch.sum((self._actions - self._prev_actions) ** 2, dim=-1)
        self._prev_actions = self._actions.clone()

        # joint velocity penalty
        joint_vel = torch.sum(self.robot.data.joint_vel[:, self._left_ids] ** 2, dim=-1)

        reward = (
            self.cfg.w_left_pos       * left_pos_err
            + self.cfg.w_left_pos_fine  * (1.0 - torch.tanh(left_pos_err  / self.cfg.pos_fine_std))
            + self.cfg.w_left_ori     * left_ori_err
            + self.cfg.w_action_rate  * action_rate
            + self.cfg.w_joint_vel    * joint_vel
        )

        return reward

    # ------------------------------------------------------------------
    # dones
    # ------------------------------------------------------------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        truncated  = self.episode_length_buf >= self.max_episode_length
        return terminated, truncated

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)

        n = len(env_ids)

        # reset joints: default_pos * U(0.5, 1.5)  (matches reset_joints_by_scale)
        joint_pos_deltas = 2. * (torch.rand_like(self.robot_start_joint_pos[env_ids]) - 0.5)
        joint_vel_deltas = 2. * (torch.rand_like(self.robot_start_joint_vel[env_ids]) - 0.5)
    
        # Calculate joint positions
        dof_pos = 0.35 * joint_pos_deltas
        dof_pos += self.robot_start_joint_pos[env_ids].clone()
        # Now clamp
        dof_pos = torch.clamp(dof_pos,min=self.robot_dof_lower_limits,max=self.robot_dof_upper_limits)

        dof_vel = 1. * joint_vel_deltas
        dof_vel += self.robot_start_joint_vel[env_ids].clone()

        self.robot.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids, joint_ids=self.actuated_dof_indices)
        
        # Reset position and velocity targets to the actual robot position and velocity
        self.robot.set_joint_position_target(dof_pos, env_ids=env_ids, joint_ids=self.actuated_dof_indices)
        self.robot.set_joint_velocity_target(dof_vel, env_ids=env_ids, joint_ids=self.actuated_dof_indices)

        # resample EE target commands
        self._resample_commands(env_ids)
        self._cmd_timer[env_ids] = 0

        # clear action buffers
        self._actions[env_ids]      = 0.0
        self._prev_actions[env_ids] = 0.0

    # ------------------------------------------------------------------
    # command sampling helpers
    # ------------------------------------------------------------------
    def _resample_commands(self, env_ids: torch.Tensor):
        n = len(env_ids)
        cfg = self.cfg

        # --- left arm ---
        lx = self._sample_uniform(cfg.left_pos_x, n)
        ly = self._sample_uniform(cfg.left_pos_y, n)
        lz = self._sample_uniform(cfg.left_pos_z, n)
        lr = self._sample_uniform(cfg.left_roll,  n)
        lp = self._sample_uniform(cfg.left_pitch, n)
        lyw = self._sample_uniform(cfg.left_yaw, n)
        lq = quat_from_euler_xyz(lr, lp, lyw)   # (N, 4) w,x,y,z

        self._left_cmd[env_ids, 0] = lx
        self._left_cmd[env_ids, 1] = ly
        self._left_cmd[env_ids, 2] = lz
        self._left_cmd[env_ids, 3:] = lq

        # # --- right arm ---
        # rx = self._sample_uniform(cfg.right_pos_x, n)
        # ry = self._sample_uniform(cfg.right_pos_y, n)
        # rz = self._sample_uniform(cfg.right_pos_z, n)
        # rr = self._sample_uniform(cfg.right_roll,  n)
        # rp = torch.full((n,), cfg.right_pitch, device=self.device)
        # ryw = self._sample_uniform(cfg.right_yaw, n)
        # rq = quat_from_euler_xyz(rr, rp, ryw)

        # self._right_cmd[env_ids, 0] = rx
        # self._right_cmd[env_ids, 1] = ry
        # self._right_cmd[env_ids, 2] = rz
        # self._right_cmd[env_ids, 3:] = rq

    def _sample_uniform(self, bounds: tuple | float, n: int) -> torch.Tensor:
        if isinstance(bounds, (int, float)):
            return torch.full((n,), float(bounds), device=self.device)
        lo, hi = bounds
        return torch.zeros(n, device=self.device).uniform_(lo, hi)
