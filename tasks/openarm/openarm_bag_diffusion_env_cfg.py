# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# 
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
import pathlib
import numpy as np
import warp as wp
import math
from dextrah_lab.assets.openarm.openarm_bimanual_tether import OPEN_ARM_CFG, OPEN_ARM_HIGH_PD_CFG

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelWithAdditiveBiasCfg

@configclass
class EventCfg:
    """Configuration for randomization."""

    # NOTE: the below ranges form the initial ranges for the parameters

    # -- robot
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (1.0, 1.0),
            "num_buckets": 250,
        },
    )

    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (1., 1.),
            "damping_distribution_params": (1., 1.),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    robot_joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": (0. , 0.),
             # NOTE: I don't really care about this one
#            "lower_limit_distribution_params": (0.00, 0.01),
#            "upper_limit_distribution_params": (0.00, 0.01),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    # -- object
    object_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object", body_names=".*"),
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (1.0, 1.0),
            "num_buckets": 250,
        },
    )
    object_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (1., 1.),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    # NOTE: I don't really care about this one
#    # -- scene
#    reset_gravity = EventTerm(
#        func=mdp.randomize_physics_scene_gravity,
#        mode="interval",
#        is_global_time=True,
#        interval_range_s=(36.0, 36.0),  # time_s = num_steps * (decimation * dt)
#        params={
#            "gravity_distribution_params": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.4]),
#            "operation": "add",
#            "distribution": "gaussian",
#        },
#    )

@configclass
class OpenarmEnvCfg(DirectRLEnvCfg):
    # Placeholder for objects_dir which targets the directory of objects for training
    objects_dir = "handbag"
    valid_objects_dir = ["handbag"]

    # Toggle for using cuda graph
    use_cuda_graph = False

    # env
    sim_dt = 1/120.
    decimation = 12 # 10 Hz
    num_sim_steps_to_render=12
    episode_length_s = 10.
    num_actions = 7
    success_timeout = 2.
    distillation = False
    num_states = 34
    num_observations = 34
    num_student_observations = 21
    num_teacher_observations = num_observations

    state_space = 0
    observation_space = 0
    action_space = 0

    #action_scale = (0.01, 0.05, 0.044)
    action_scale = (0.015, 0.075, 0.044)
    #action_scale = (0.02, 0.1, 0.044)
    
    asymmetric_obs = True
    obs_type = "full"
    simulate_stereo = False
    stereo_baseline = 55 / 1000

    # viewer camera
    viewer: ViewerCfg = ViewerCfg(
        eye=(11.0, 0., 3.5),
        lookat=(0.0, 0., -3.),
    )

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=sim_dt,
        render_interval=num_sim_steps_to_render,
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
            gpu_max_rigid_patch_count=4 * 5 * 2**15,
            gpu_collision_stack_size=2**27,
        ),
    )
    # robot
    robot_cfg: ArticulationCfg = OPEN_ARM_HIGH_PD_CFG.replace(prim_path="/World/envs/env_.*/Robot").replace(
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                    "openarm_left_joint1": 1.3,
                    "openarm_left_joint2": -0.35,
                    "openarm_left_joint3": -0.24,
                    "openarm_left_joint4": 2.3,
                    "openarm_left_joint5": 1.,
                    "openarm_left_joint6": 0.,

                    "openarm_right_joint1": -1.3,
                    "openarm_right_joint2": 0.35,
                    "openarm_right_joint3": 0.24,
                    "openarm_right_joint4": 2.3,
                    "openarm_right_joint5": -1.,
                    "openarm_right_joint6": 0.,

                    ".*mcp_flex": 0.,
                    ".*pip": 0.,
                    ".*dip": 0.,
                    ".*cmc_abd": 0.,
                    ".*cmc_flex": 0.,
                    ".*mcp": 0.,
                    ".*_ip": 0.,
                }, 
        )
    )
    left_arm_joint_name = [
            "openarm_left_joint1", "openarm_left_joint2", "openarm_left_joint3",
            "openarm_left_joint4", "openarm_left_joint5", "openarm_left_joint6",
    ]
    right_arm_joint_name = [
        "openarm_right_joint1", "openarm_right_joint2", "openarm_right_joint3",
        "openarm_right_joint4", "openarm_right_joint5", "openarm_right_joint6",
    ]

    left_gripper_joint_name = ["left_index_mcp_flex", "left_index_pip", "left_index_dip", 
                               "left_middle_mcp_flex", "left_middle_pip", "left_middle_dip",
                               "left_pinky_mcp_flex", "left_pinky_pip", "left_pinky_dip",
                                 "left_ring_mcp_flex", "left_ring_pip", "left_ring_dip",
                               "left_thumb_cmc_abd", "left_thumb_cmc_flex", "left_thumb_mcp", "left_thumb_ip"]
    
    right_gripper_joint_name = ["right_index_mcp_flex", "right_index_pip", "right_index_dip", 
                                "right_middle_mcp_flex", "right_middle_pip", "right_middle_dip", 
                                "right_pinky_mcp_flex", "right_pinky_pip", "right_pinky_dip",
                                "right_ring_mcp_flex", "right_ring_pip", "right_ring_dip",
                                "right_thumb_cmc_abd", "right_thumb_cmc_flex", "right_thumb_mcp", "right_thumb_ip"]

    left_tcp_name = "openarm_left_ee_tcp"
    right_tcp_name = "openarm_right_ee_tcp"

    module_path = os.path.dirname(__file__)
    root_path = os.path.dirname(os.path.dirname(module_path))
    scene_objects_usd_path = os.path.join(root_path, "assets/scene_objects/")
    primitives_usd_path = os.path.join(root_path, "assets/primitives/USD/")

    dome_light_dir = os.path.join(
        root_path, "assets", "dome_light_textures"
    )
    # table_texture_dir = os.path.join(
    #     root_path, "assets", "curated_table_textures"
    # )
    # metropolis_asset_dir = os.path.join(
    #     root_path, "assets", "object_textures"
    # )

    table_texture_dir = os.path.join(
        root_path, "assets", "background_imgs/voc_resized"
    )
    metropolis_asset_dir = os.path.join(
        root_path, "assets",  "perlin_noise"
    )

    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=primitives_usd_path + "large_8_cuboid/large_8_cuboid.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1000.0,
            ),
            scale=(1., 1., 1.),
            mass_props=sim_utils.MassPropertiesCfg(density=30.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.5, 0., 0.5),
            rot=(1.0, 0.0, 0.0, 0.0)),
    )

    # table
    table_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=scene_objects_usd_path + "table.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
            ),
            scale=(2.2, 1.73, 1.),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            # pos=(0.05 + 0.725 / 2,
            #      0.668 - 1.16 / 2,
            #      0.21 - 0.03 / 2),
            pos=(0.7,
                 0.,
                 0.20 - 0.03 / 2),
            rot=(1.0, 0.0, 0.0, 0.0)), 
    )
    # distillation related parameters
    img_width = 120  
    img_height = 96
    # img_width = 240  
    # img_height = 192

    distil_depth_width = 120
    distil_depth_height = 96

    camera_rand_rot_range = 3
    camera_rand_pos_range = 0.03

    # head cam
    head_camera_pos = [0.10, 0.0, 0.52]
    head_camera_rot = [0.68301, 0.18301, -0.18301, -0.68301]
    head_img_width = img_width     
    head_img_height = img_height
    # head_img_width = 240  
    # head_img_height = 192           
    fps = 30.0

    hfov = float(np.deg2rad(115))         # Horizontal field of view
    vfov = float(np.deg2rad(90.3))          # Vertical field of view
    efl_mm = 3.          # Effective focal length

    FX = float((head_img_width  / 2.0) / np.tan(hfov / 2.0))
    FY = float((head_img_height / 2.0) / np.tan(vfov / 2.0))
    CX = head_img_width  / 2.0
    CY = head_img_height / 2.0

    head_cam_intrinsic_matrix = [[FX, 0.0, CX], 
                                 [0.0, FY, CY], 
                                 [0.0, 0.0, 1.0]]

    horizontal_aperture = float(2.0 * efl_mm * np.tan(hfov / 2.0))
    vertical_aperture = float(2.0 * efl_mm * np.tan(vfov / 2.0))

    head_cam_cfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/openarm_body_link/Camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=head_camera_pos,
            rot=head_camera_rot,
            convention="usd",
        ),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=efl_mm,                       
            horizontal_aperture=horizontal_aperture, 
            vertical_aperture=vertical_aperture,   
            clipping_range=(0.01, 5.),   
            projection_type="pinhole",
        ),
        width=head_img_width,   
        height=head_img_height,   
        data_types=["rgb", "depth",],
        #update_period=1.0 / fps,  
    )   

    # wrist cam
    #wrist_camera_pos = [0.1, 0., 0.1201]
    wrist_camera_pos = [-0.06, 0.0127, 0.06]
    wrist_camera_rot = [0.1496, 0.15142, -0.72721, -0.65257]
    wrist_img_width = img_width      
    wrist_img_height = img_height   
    # wrist_img_width = 240  
    # wrist_img_height = 192         
    fps = 30.0

    hfov = float(np.deg2rad(98.96))         # Horizontal field of view
    vfov = float(np.deg2rad(77.24))          # Vertical field of view
    efl_mm = 3.55          # Effective focal length

    FX = float((wrist_img_width  / 2.0) / np.tan(hfov / 2.0))
    FY = float((wrist_img_height / 2.0) / np.tan(vfov / 2.0))
    CX = wrist_img_width  / 2.0
    CY = wrist_img_height / 2.0

    wrist_cam_intrinsic_matrix = [[FX, 0.0, CX], 
                                  [0.0, FY, CY], 
                                  [0.0, 0.0, 1.0]]

    horizontal_aperture = float(2.0 * efl_mm * np.tan(hfov / 2.0))
    vertical_aperture = float(2.0 * efl_mm * np.tan(vfov / 2.0))

    wrist_L_cam_cfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/left_hand/left_base_link/Camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=wrist_camera_pos,
            rot=wrist_camera_rot,
            convention="usd",
        ),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=efl_mm,                       
            horizontal_aperture=horizontal_aperture, 
            vertical_aperture=vertical_aperture,   
            clipping_range=(0.01, 5.),   
            projection_type="pinhole",
        ),
        width=wrist_img_width,   
        height=wrist_img_height,   
        data_types=["rgb", "depth",],
        #update_period=1.0 / fps,  
    )

    wrist_R_cam_cfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/right_hand/right_base_link/Camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=wrist_camera_pos,
            rot=wrist_camera_rot,
            convention="usd",
        ),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=efl_mm,                       
            horizontal_aperture=horizontal_aperture, 
            vertical_aperture=vertical_aperture,   
            clipping_range=(0.01, 5.),   
            projection_type="pinhole",
        ),
        width=wrist_img_width,   
        height=wrist_img_height,   
        data_types=["rgb", "depth",],
        #update_period=1.0 / fps,  
    )

    pred_pos_marker_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/pos_marker",
        markers={
            "goal": sim_utils.SphereCfg(
                radius=0.01,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            )
        },
    )

    gt_pos_marker_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/pos_marker",
        markers={
            "goal": sim_utils.SphereCfg(
                radius=0.01,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            )
        },
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2., replicate_physics=False)

    # reward weights
    hand_to_object_weight = 1.
    hand_to_object_sharpness = 15.
    object_to_goal_weight = 5.
    in_success_region_at_rest_weight = 10.
    lift_sharpness = 8.5

    # Goal reaching parameters
    object_goal_tol = 0.02 # m
    success_for_adr = 0.2
    min_steps_for_dr_change = 240 # number of steps
    #min_steps_for_dr_change = 5 * int(episode_length_s / (decimation * sim_dt))

    # Lift criteria
    min_num_episode_steps = 60
    object_height_thresh = 0.15

    # Object spawning params
    x_center = 0.25
    x_width = 0.1
    y_center = 0.2
    y_width = 0.15

    # DR Controls
    enable_adr = True
    num_adr_increments = 50
    starting_adr_increments = 0 # 0 for no DR up to num_adr_increments for max DR 

    # Default dof friction coefficients
    # NOTE: these are set based on how far out they will scale multiplicatively
    # with the above robot_joint_friction EventTerm above.
    starting_robot_dof_friction_coefficients = [
        1., 1., 1., 1.,
        1., 1., 1., 1., 
        1., 1., 1., 1., 
        1., 1., 1., 1., 
        1., 1., 1., 1.,
        1., 1., 1., 1.,
        1., 1., 1., 1., 
        1., 1., 1., 1., 
        1., 1., 1., 1., 
        1., 1., 1., 1.,
        1., 1., 1., 1.,
        1., 1.,
    ]

    # domain randomization config
    events: EventCfg = EventCfg()

    # These serve to set the maximum value ranges for the different physics parameters
    adr_cfg_dict = {
        "num_increments": num_adr_increments, # number of times you can change the parameter ranges
        "robot_physics_material": {
            "static_friction_range": (0.5, 1.2),
            "dynamic_friction_range": (0.3, 1.0),
            "restitution_range": (0.8, 1.0)
        },
        "robot_joint_stiffness_and_damping": {
            "stiffness_distribution_params": (0.5, 2.),
            "damping_distribution_params": (0.5, 2.),
        },
        "robot_joint_friction": {
            "friction_distribution_params": (0., 5.),
        },
        "object_physics_material": {
            "static_friction_range": (0.5, 1.2),
            "dynamic_friction_range": (0.3, 1.0),
            "restitution_range": (0.8, 1.0)
        },
        "object_scale_mass": {
            "mass_distribution_params": (0.5, 3.),
        },
    }

    # Object disturbance wrench fixed params
    wrench_trigger_every = int(1. / (decimation * sim_dt)) # 1 sec
    torsional_radius = 0.01 # m
    hand_to_object_dist_threshold = 0.2 # m
    #wrench_prob_per_rollout = 0. # NOTE: currently not used

    # Object scaling
    object_scale_max = 1.1
    object_scale_min = 0.9
    deactivate_object_scaling = True

    aux_coeff = 10.

    # Dictionary of custom parameters for ADR
    # NOTE: first number in range is the starting value, second number is terminal value
    adr_custom_cfg_dict = {
        "object_wrench": {
            "max_linear_accel": (0., 10.)
        },
        "object_spawn": {
            "x_width_spawn": (0., x_width),
            "y_width_spawn": (0., y_width),
            "rotation": (0., 1.)
        },
        "object_state_noise": {
            "object_pos_noise": (0.0, 0.03), # m
            "object_pos_bias": (0.0, 0.02), # m
            "object_rot_noise": (0.0, 0.1), # rad
            "object_rot_bias": (0.0, 0.08), # rad
        },
        "robot_spawn": {
            "joint_pos_noise": (0., 0.1),
            "joint_vel_noise": (0., 0.)
        },
        "robot_state_noise": {
            "robot_joint_pos_noise": (0.0, 0.1), # rad
            "robot_joint_pos_bias": (0.0, 0.1), # rad
            "robot_joint_vel_noise": (0.0, 0.18), # rad
            "robot_joint_vel_bias": (0.0, 0.08), # rad
        },
        "reward_weights": {
            "finger_curl_reg": (-0.01, -0.01),
            "object_to_goal_sharpness": (-150., -155.),
            "lift_weight": (5., 0.)
        },
        "pd_targets": {
            "velocity_target_factor": (1., 0.)
        },
        "fabric_damping": {
            "gain": (10., 20.)
        },
        "observation_annealing": {
            "coefficient": (0., 0.)
        },
    }

    # depth randomization parameters
    img_aug_type = "rgb"
    aug_depth = True
    cam_matrix = wp.mat44f()
    cam_matrix[0,0] = 2.2460368
    cam_matrix[1, 1] = 2.9947157
    cam_matrix[2, 3] = -1.
    cam_matrix[3, 2] = 1.e-3
    d_min = 0.01
    d_max = 1.
    depth_randomization_cfg_dict = {
        # Dropout and random noise blob parameters
        "pixel_dropout_and_randu": {
            "p_dropout": 0.0125 / 4,
            "p_randu": 0.0125 / 4,
            "d_max": d_min,
            "d_min": d_max,
        },
        # Random stick parameters
        "sticks": {
            "p_stick":  0.001 / 4,
            "max_stick_len": 18.,
            "max_stick_width": 3.,
            "d_max": d_min,
            "d_min": d_max,
        },
        # Correlated noise parameters
        "correlated_noise": {
            "sigma_s": 1./2,
            "sigma_d": 1./6,
            "d_max": d_min,
            "d_min": d_max,
        },
        # Normal noise parameters
        "normal_noise": {
            "sigma_theta": 0.01,
            "cam_matrix": cam_matrix,
            "d_max": d_min,
            "d_min": d_max,
        }

    }

    # If enabled, the environment will terminate when time out.
    # This is used for data recording.
    disable_out_of_reach_done = False
    disable_dome_light_randomization = False
    disable_arm_randomization = False
    simulate_stereo = True
    # Enable depth image for teacher RL training (end-to-end vision-based RL)
    use_depth_teacher = False
