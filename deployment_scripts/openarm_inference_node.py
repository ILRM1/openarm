# System imports
import os
import sys
import pathlib
from threading import Lock, Thread
import copy
import argparse

# ROS imports
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TransformStamped
from sensor_msgs.msg import JointState, Image
from tf2_ros import TransformBroadcaster
from std_msgs.msg import Bool
from rclpy.qos import qos_profile_sensor_data
import message_filters

# Third party
import torch
import yaml
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
import pytorch_kinematics as pk
from typing import Optional

# CV
from cv_bridge import CvBridge
import cv2

# RL games
from rl_games.algos_torch import model_builder
from rl_games.algos_torch.model_builder import ModelBuilder
from rl_games.algos_torch import torch_ext

# Dextrah FGP
from dextrah_lab.distillation.a2c_stereo_transformer import A2CBuilder as A2CStereoTransformerBuilder

#cv2.namedWindow('Left RGB Image', cv2.WINDOW_AUTOSIZE)
#cv2.namedWindow('Right RGB Image', cv2.WINDOW_AUTOSIZE)

def load_param_dict(cfg_path):
    with open(cfg_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def adjust_state_dict_keys(checkpoint_state_dict, model_state_dict):
    adjusted_state_dict = {}
    num_elems = 0
    for key, value in checkpoint_state_dict.items():
        num_elems += value.numel()
        # If the key is in the model's state_dict, use it directly
        if key in model_state_dict:
            adjusted_state_dict[key] = value
        else:
            # Try inserting '_orig_mod' in different positions based on key structure
            parts = key.split(".")
            new_key_with_orig_mod = None

            # Try inserting '_orig_mod' before the last layer index for different cases
            parts.insert(2, "_orig_mod")
            new_key_with_orig_mod = ".".join(parts)

            # If adding '_orig_mod' matches a key in the model, use the modified key
            if new_key_with_orig_mod in model_state_dict:
                adjusted_state_dict[new_key_with_orig_mod] = value
            else:
                # check if removing orig_mod works
                key_no_orig_mod = key.replace("_orig_mod.", "")
                if key_no_orig_mod in model_state_dict:
                    adjusted_state_dict[key_no_orig_mod] = value
                else:
                    # Log the key that couldn't be matched, for debugging purposes
                    print(f"Could not match key: {key} -> {new_key_with_orig_mod}")
                    # If neither works, retain the original key as a fallback
                    adjusted_state_dict[key] = value

    print(f"Number of elements in adjusted state dict: {num_elems}")
    return adjusted_state_dict


class DextrAHFGP:
    def __init__(
        self, cfg_path, img_shape, num_proprio_obs,
        num_actions, ckpt_path=None, device="cuda"
    ):
        self.cfg_path = cfg_path
        self.ckpt_path = ckpt_path
        self.device = device

        # read the yaml file
        network_params = load_param_dict(cfg_path)["params"]
        self.num_proprio_obs = num_proprio_obs
        self.img_shape = img_shape

        # build the model config
        normalize_value = network_params["config"]["normalize_value"]
        normalize_input = network_params["config"]["normalize_input"]
        model_config = {
            "actions_num": num_actions,
            "input_shape": (num_proprio_obs,),
            "num_seqs": 2,
            "value_size": 1,
            'normalize_value': normalize_value,
            'normalize_input': normalize_input,
            'num_envs': 2,
        }

        # build the model
        builder = ModelBuilder()
        network = builder.load(network_params)
        self.model = network.build(model_config).to(self.device)
        self.model.eval()

        # load checkpoint if available
        if ckpt_path is not None:
            weights = torch_ext.load_checkpoint(ckpt_path)
            weights["model"] = adjust_state_dict_keys(
                weights["model"],
                self.model.state_dict()
            )
            self.model.load_state_dict(weights["model"])
            if normalize_input and 'running_mean_std' in weights:
                self.model.running_mean_std.load_state_dict(
                    weights["running_mean_std"]
                )
        # self.model = torch.compile(self.model)

        if self.model.is_rnn():
            hidden_states = self.model.get_default_rnn_state()
            self.hidden_states = [
                s.to(self.device) for s in hidden_states
            ]

        # dummy variable, this doesn't actually contain prev actions
        # need this bc of rl_games weirdness...
        self.dummy_prev_actions = torch.zeros(
            (2, num_actions), dtype=torch.float32
        ).to(self.device)


    def step(self, proprio, left_img, right_img, hidden_states=None):
        # package observations
        batch_dict = {
            "is_train": True,
            "obs": proprio.repeat(2, 1),
            "img_left": left_img.repeat(2, 1, 1, 1),
            "img_right": right_img.repeat(2, 1, 1, 1),
            "prev_actions": self.dummy_prev_actions,
            "finetune_backbone": False
        }
        if hidden_states is None:
            hidden_states = self.hidden_states
        # add extra information for RNNs
        if self.model.is_rnn():
            batch_dict["rnn_states"] = hidden_states
            batch_dict["seq_length"] = 1
            batch_dict["rnn_masks"] = None

        # step through model
        with torch.no_grad():
            res_dict = self.model(batch_dict)
        mus = res_dict["mus"][0:1]
        sigmas = res_dict["sigmas"][0:1]

        self.hidden_states = [
            s for s in res_dict["rnn_states"][0]
        ]

        position = res_dict["rnn_states"][1]['object_pos'][0:1]

        selected_action = mus

        return {
            "mus": mus,
            "sigmas": sigmas,
            "obj_pos": position,
            "selected_action": selected_action
        }


    def setup_cuda_graph(self):
        dummy_proprio = torch.randn(1, self.num_proprio_obs).to(self.device)
        dummy_left_img = torch.randn(1, *self.img_shape).to(self.device)
        dummy_right_img = torch.randn(1, *self.img_shape).to(self.device)

        for _ in range(3):
            self.step(
                dummy_proprio, dummy_left_img, dummy_right_img
            )

        self.reset_hidden_state()
        self.cuda_graph = torch.cuda.CUDAGraph()

        self.static_proprio = torch.empty_like(dummy_proprio, device=self.device)
        self.static_left_img = torch.empty_like(dummy_left_img, device=self.device)
        self.static_right_img = torch.empty_like(dummy_right_img, device=self.device)
        self.hidden_state_1 = torch.empty_like(self.hidden_states[0], device=self.device)
        self.hidden_state_2 = torch.empty_like(self.hidden_states[1], device=self.device)
        with torch.cuda.graph(self.cuda_graph):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                self._policy_out = self.step(
                    self.static_proprio,
                    self.static_left_img,
                    self.static_right_img,
                    [self.hidden_state_1, self.hidden_state_2]
                )

    def step_cuda_graph(self, proprio, left_img, right_img):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            self.static_proprio.copy_(proprio)
            self.static_left_img.copy_(left_img)
            self.static_right_img.copy_(right_img)
            self.hidden_state_1.copy_(self.hidden_states[0])
            self.hidden_state_2.copy_(self.hidden_states[1])
            self.cuda_graph.replay()

            policy_out = self._policy_out
            policy_out = {
                "mus": self._policy_out["mus"].clone(),
                "sigmas": self._policy_out["sigmas"].clone(),
                "obj_pos": self._policy_out["obj_pos"].clone(),
            }
            try:
                policy_out["selected_action"] = torch.distributions.Normal(
                    policy_out["mus"],
                    policy_out["sigmas"]
                ).sample()
            except Exception as e:
                breakpoint()

        return policy_out

    def reset_hidden_state(self):
        for i in range(len(self.hidden_states)):
            self.hidden_states[i].zero_()

class DextrahFGPNode(Node):
    def __init__(self, node_name: str) -> None:
        super().__init__(node_name)

        # Set up cuda and warp
        self.device='cuda'
        self.batch_size = 1
        self.num_actions = 7
        self.num_obs = 24
        self.action_scale = (0.02, 0.1, 1.)
        self.tcp_pose_min = torch.tensor([0.05, 0., 0.22, -np.pi*2, -np.pi*2, -np.pi*2], device=self.device)
        self.tcp_pose_max = torch.tensor([0.45, 0.4, 0.35, np.pi*2, np.pi*2, np.pi*2], device=self.device)
        self.student_ckpt = "distillation/runs/dextrah_student_20000_iters.pth"

        # For converting ROS image messages to CV formates
        self.bridge = CvBridge()

        # Camera / robot state
        self._synced_lock = Lock()
        self._left_image = None
        self._right_image = None
        self._image_height = 384
        self._image_width = 480
        self._downsample_factor = 2
        self.synced_feedback_time = time.time()

        self.robot_q = torch.zeros(self.batch_size, 16, device=self.device)
        self.robot_qd = torch.zeros(self.batch_size, 16, device=self.device)
        self.left_tcp_pose = torch.zeros(self.batch_size, 6, device=self.device)

        # ApproximateTimeSynchronizer: camera(L) + camera(R) + joint_states + tcp_pose
        _left_img_sub  = message_filters.Subscriber(self, Image,       '/camera/image',      qos_profile=qos_profile_sensor_data)
        _right_img_sub = message_filters.Subscriber(self, Image,       '/wristcam_left/image', qos_profile=qos_profile_sensor_data)
        _joint_sub     = message_filters.Subscriber(self, JointState,  '/joint_states')
        _tcp_sub       = message_filters.Subscriber(self, PoseStamped, '/tcp_pose/left')
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [_left_img_sub, _right_img_sub, _joint_sub, _tcp_sub],
            queue_size=10,
            slop=0.01,
        )
        self._sync.registerCallback(self._synced_callback)

        self._publish_rate = 10. # Hz
        self._publish_dt = 1./self._publish_rate # s

        # Goal to bring object to. NOTE: this should be a command in the future
        self.object_goal = torch.tensor([0.25, 0.15, 0.3], device=self.device).repeat((self.batch_size, 1))

        self.last_actions = torch.zeros(self.batch_size, self.num_actions, device=self.device)

        # publishers
        # For commanding pose target
        self.left_tcp_pose_targets = None
        self._left_tcp_pose_lock = Lock()
        self._left_tcp_pose_command_pub = self.create_publisher(PoseStamped, '/left/ik_target_pose', 1)
        self._left_tcp_pose_timer = self.create_timer(self._publish_dt, self._left_tcp_target_pose_pub_callback)

        self.left_gripper_pos_targets = None
        self._left_gripper_pos_lock = Lock()

        self._gripper_pos_command_pub = self.create_publisher(JointState, '/left/gripper_command', 1)
        self._left_gripper_pos_timer = self.create_timer(self._publish_dt, self._left_gripper_pos_pub_callback)
        self._arm_init_pub = self.create_publisher(JointState, '/arm/command', 1)

        # Create publisher for predicted object pose
        self._object_pos_lock = Lock()
        self.object_pos = None
        # self._object_pos_pub = self.create_publisher(PoseStamped, "/kuka_allegro_fabric/predicted_obj_pose", 1)
        self.object_pos_tf = TransformBroadcaster(self)
        self._fgp_object_pos_timer = self.create_timer(self._publish_dt, self._object_pos_callback)

        # Instantiate FGP
        self.init_fgp()

    def init_fgp(self):
        # get path to config file
        #parent_path = str(pathlib.Path(__file__).parent.parent.parent.resolve())
        parent_path = str(pathlib.Path(__file__).parent.resolve())
        parent_path = parent_path.replace("deployment_scripts", "")
        agent_cfg_folder = "tasks/dextrah_kuka_allegro/agents"
        student_cfg_path = os.path.join(
            parent_path,
            agent_cfg_folder,
            #"rl_games_ppo_lstm_scratch_cnn_aux_stereo.yaml",
            "rl_games_ppo_stereo_transformer.yaml"
        )

        # get path to checkpoint
        student_ckpt_path = os.path.join(parent_path, self.student_ckpt)

        # register our custom model with the rl_games model builder
        model_builder.register_network("a2c_stereo_transformer", A2CStereoTransformerBuilder)

        img_shape = (3, self._image_height//self._downsample_factor, self._image_width//self._downsample_factor)
        # create the model
        self.dextrah_fgp = DextrAHFGP(
            cfg_path=student_cfg_path,
            img_shape=img_shape,
            num_proprio_obs=self.num_obs,
            num_actions=self.num_actions,
            ckpt_path=student_ckpt_path,
            device=self.device
        )

        # Reset hidden state
        self.dextrah_fgp.reset_hidden_state()

        # Perform cuda graph capture of FGP
        self.dextrah_fgp.setup_cuda_graph()

    def _img_to_tensor(self, msg):
        img_np = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8').astype(np.float32)
        img_np = cv2.resize(
            img_np,
            (self._image_width // self._downsample_factor, self._image_height // self._downsample_factor),
            interpolation=cv2.INTER_LINEAR,
        )
        img_np = np.transpose(img_np, (2, 0, 1)) / 255.
        return torch.from_numpy(img_np).to(self.device).unsqueeze(0)

    def _synced_callback(self, left_img_msg, right_img_msg, joint_msg, tcp_msg):
        # Images
        left_img  = self._img_to_tensor(left_img_msg)
        right_img = self._img_to_tensor(right_img_msg)

        # Joint state (positional: right[0-6], right_finger[7], left[8-14], left_finger[15])
        pos = joint_msg.position
        vel = joint_msg.velocity if joint_msg.velocity else [0.0] * len(pos)
        r_pos  = list(pos[0:7]);   r_vel  = list(vel[0:7])
        rf_pos = [pos[7]];         rf_vel = [vel[7]]
        l_pos  = list(pos[8:15]);  l_vel  = list(vel[8:15])
        lf_pos = [pos[15]];        lf_vel = [vel[15]]

        robot_q  = torch.zeros(self.batch_size, 16, device=self.device)
        robot_qd = torch.zeros(self.batch_size, 16, device=self.device)
        robot_q[:,  :7]   = torch.tensor([l_pos],  dtype=torch.float32, device=self.device)
        robot_qd[:, :7]   = torch.tensor([l_vel],  dtype=torch.float32, device=self.device)
        robot_q[:,  7:14] = torch.tensor([r_pos],  dtype=torch.float32, device=self.device)
        robot_qd[:, 7:14] = torch.tensor([r_vel],  dtype=torch.float32, device=self.device)
        robot_q[:,  14:15] = torch.tensor([lf_pos], dtype=torch.float32, device=self.device)
        robot_qd[:, 14:15] = torch.tensor([lf_vel], dtype=torch.float32, device=self.device)
        robot_q[:,  15:16] = torch.tensor([rf_pos], dtype=torch.float32, device=self.device)
        robot_qd[:, 15:16] = torch.tensor([rf_vel], dtype=torch.float32, device=self.device)

        # TCP pose
        p = tcp_msg.pose.position
        o = tcp_msg.pose.orientation
        euler = R.from_quat([o.x, o.y, o.z, o.w]).as_euler('xyz')
        left_tcp_pose = torch.tensor(
            [[p.x, p.y, p.z, euler[0], euler[1], euler[2]]],
            dtype=torch.float32, device=self.device,
        )

        with self._synced_lock:
            self.synced_feedback_time = time.time()
            self._left_image  = left_img
            self._right_image = right_img
            self.robot_q.copy_(robot_q)
            self.robot_qd.copy_(robot_qd)
            self.left_tcp_pose.copy_(left_tcp_pose)

    def _left_tcp_target_pose_pub_callback(self):
        """
        Publishes latest FGP pose command.
        """
        left_tcp_pose_targets = None
        with self._left_tcp_pose_lock:
            if self.left_tcp_pose_targets is not None:
                left_tcp_pose_targets = self.left_tcp_pose_targets[0,:].float().detach().cpu().numpy()
        
        if left_tcp_pose_targets is not None:
            msg = PoseStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'openarm_body_link0'
            msg.pose.position.x = float(left_tcp_pose_targets[0])
            msg.pose.position.y = float(left_tcp_pose_targets[1])
            msg.pose.position.z = float(left_tcp_pose_targets[2])
            msg.pose.orientation.w = float(left_tcp_pose_targets[3])
            msg.pose.orientation.x = float(left_tcp_pose_targets[4])
            msg.pose.orientation.y = float(left_tcp_pose_targets[5])
            msg.pose.orientation.z = float(left_tcp_pose_targets[6])
            self._left_tcp_pose_command_pub.publish(msg)

    def _left_gripper_pos_pub_callback(self):
        """
        Publishes latest FGP pca command.
        """
        left_gripper_pos_targets = None
        with self._left_gripper_pos_lock:
            if self.left_gripper_pos_targets is not None:
                left_gripper_pos_targets = self.left_gripper_pos_targets.float().detach().cpu().numpy()
      
        if left_gripper_pos_targets is not None:
      
            msg = JointState()
            msg.name = ['openarm_left_finger_joint1']
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.position = left_gripper_pos_targets.tolist()
            msg.velocity = []
            msg.effort = []
            self._gripper_pos_command_pub.publish(msg)

    def publish_init_pos(self):
        joint_names = [
            'openarm_right_joint1',
            'openarm_right_joint2',
            'openarm_right_joint3',
            'openarm_right_joint4',
            'openarm_right_joint5',
            'openarm_right_joint6',
            'openarm_right_joint7',
            'openarm_right_finger_joint1',
            'openarm_left_joint1',
            'openarm_left_joint2',
            'openarm_left_joint3',
            'openarm_left_joint4',
            'openarm_left_joint5',
            'openarm_left_joint6',
            'openarm_left_joint7',
            'openarm_left_finger_joint1',
        ]
        target = [-0.9, 0.35, 0.24, 2.0, 0.54, 0., -1.1, 1.,
                  0.9, -0.35, -0.24, 2.0, -0.54, 0., 1.1, 1.]
        
        # robot_q layout: [left_joints(0-6), right_joints(7-13), left_finger(14), right_finger(15)]
        # msg.position layout: [left_joints(0-6), left_finger(7), right_joints(8-14), right_finger(15)]
        with self._synced_lock:
            q = self.robot_q[0].cpu().numpy()
        current = [q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[14],
                   q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[15]]

        num_steps = 100
        for step in range(1, num_steps + 1):
            alpha = step / num_steps
            interp = [c + alpha * (t - c) for c, t in zip(current, target)]

            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = joint_names
            msg.position = interp
            msg.velocity = []
            msg.effort = []
            self._arm_init_pub.publish(msg)
            time.sleep(0.05)

        self.get_logger().info('Published init position')

    def _object_pos_callback(self):
        """
        Publishes the latest predicted object position
        """
        object_pos = None
        with self._object_pos_lock:
            if self.object_pos is not None:
                object_pos = self.object_pos[0,:].float().detach().cpu().numpy()

        if object_pos is not None:
      
#            pose_msg = PoseStamped()
#            pose_msg.header.stamp = self.get_clock().now().to_msg()
#            pose_msg.header.frame_id = 'robot_base'# Set the frame ID
#
#            pose_msg.pose.position.x = float(object_pos[0])
#            pose_msg.pose.position.y = float(object_pos[1])
#            pose_msg.pose.position.z = float(object_pos[2])
#            pose_msg.pose.orientation.x = 0.0
#            pose_msg.pose.orientation.y = 0.0
#            pose_msg.pose.orientation.z = 0.0
#            pose_msg.pose.orientation.w = 1.0
#
#            # Publish the message
#            self._object_pos_pub.publish(pose_msg)
            # Create the transform message
            t = TransformStamped()

            # Set the header information
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = 'robot_base'
            t.child_frame_id = 'obj_pos'

            # Set the translation (x, y, z)
            t.transform.translation.x = float(object_pos[0])
            t.transform.translation.y = float(object_pos[1])
            t.transform.translation.z = float(object_pos[2])

            # Set the rotation as a quaternion (x, y, z, w)
            t.transform.rotation.x = 0.0
            t.transform.rotation.y = 0.0
            t.transform.rotation.z = 0.0
            t.transform.rotation.w = 1.0

            # Broadcast the transform
            self.object_pos_tf.sendTransform(t)


    def compute_fgp_observation(self):
        with self._synced_lock:
            end = time.time()
            feedback_timed_out = (end - self.synced_feedback_time) > (3. * self._publish_dt)
            if feedback_timed_out:
                print('no synced feedback')

            robot_q      = self.robot_q[:, :7].clone()
            gripper_q    = self.robot_q[:, 14].unsqueeze(-1).clone().abs() * 0.044
            left_tcp_pose = self.left_tcp_pose.clone()
            left_image   = self._left_image.clone()  if self._left_image  is not None else None
            right_image  = self._right_image.clone() if self._right_image is not None else None

        if left_image is None or right_image is None:
            feedback_timed_out = True

        state = torch.cat(
            (robot_q, gripper_q, left_tcp_pose, self.object_goal, self.last_actions),
            dim=-1,
        )

        return state, left_image, right_image, feedback_timed_out

    def compute_actions(self, state, left_image, right_image, transmit=True, save_pth=False):
        # with torch.no_grad():
        #     action_dict = self.dextrah_fgp.step(state, left_image, right_image)

        action_dict = self.dextrah_fgp.step_cuda_graph(state, left_image, right_image)

        # NOTE: pulling out mean action. could use the selected_action
        # instead if you want a stochastic policy
        actions = action_dict["mus"]
        #actions = action_dict["selected_action"]

        # Now clip the actions between -1 and 1
        actions = torch.clamp(actions, min=-1, max=1)

        has_nan = torch.isnan(actions).any()
        has_inf = torch.isinf(actions).any()
        
        if has_nan:
            print('NaNing!!!')
        if has_inf:
            print('Infing!!!')

        object_pos = action_dict["obj_pos"]

        #print(object_pos)

        left_tcp_pose = state[:, 8:14].clone()

        left_tcp_pos_targets = left_tcp_pose[:,:3] + actions[:,:3] * self.action_scale[0]
        left_tcp_pos_targets = torch.max(torch.min(left_tcp_pos_targets, self.tcp_pose_max[:3]), self.tcp_pose_min[:3])

        # 시뮬과 동일: axis-angle delta → quat_mul(delta, curr)
        euler_np = left_tcp_pose[0, 3:6].float().cpu().detach().numpy()
        euler_corrected = euler_np.copy()
        euler_corrected[0] *= -1.
        euler_corrected[2] *= -1.
        curr_rot = R.from_euler('xyz', euler_corrected)

        delta_rotvec = (actions[0, 3:6] * self.action_scale[1]).float().cpu().detach().numpy()
        delta_rot = R.from_rotvec(delta_rotvec)
        target_rot = delta_rot * curr_rot
        tq = target_rot.as_quat()  # xyzw
        target_quat = torch.tensor([[tq[3], tq[0], tq[1], tq[2]]], dtype=torch.float32, device=self.device)
        
        left_tcp_pose_targets = torch.cat((left_tcp_pos_targets, target_quat), dim=-1)

        #left_tcp_pose_targets[:,:3] = torch.tensor([[0.3, 0.2, 0.29]], dtype=torch.float32, device=self.device)
     
        left_gripper_action = 0.5 * (actions[:,6].clone() + 1.)
        left_gripper_pos_target = torch.where(left_gripper_action>0.5, self.action_scale[2], 0.)

        # Update the action target tensors
        if transmit:
            with self._left_tcp_pose_lock:
                if self.left_tcp_pose_targets is None:
                    self.left_tcp_pose_targets = left_tcp_pose_targets
                else:
                    # In-place update to palm pose targets
                    self.left_tcp_pose_targets.copy_(left_tcp_pose_targets)

            with self._left_gripper_pos_lock: 
                if self.left_gripper_pos_targets is None:
                    self.left_gripper_pos_targets = left_gripper_pos_target
                else:
                    # In-place update to hand PCA targets
                    self.left_gripper_pos_targets.copy_(left_gripper_pos_target)            

            with self._object_pos_lock:
                if self.object_pos is None:
                    self.object_pos = object_pos.clone()
                else:
                    self.object_pos.copy_(object_pos)

            # Update last action
            self.last_actions = actions.clone()

    def burn_in(self):
        feedback_timed_out = True
        for i in range(5):
            # Pack and prepare observations
            state, left_image, right_image, feedback_timed_out = self.compute_fgp_observation()

            # Query the FGP for actions
            # NOTE: publisher callbacks will pull action data from tensors
            # into lists themselves and publish
            if not feedback_timed_out:
                print('copmute actions')
                self.compute_actions(state, left_image, right_image)
            else:
                print('not computingn actions')
                self.compute_actions(state, left_image, right_image, transmit=False)

            time.sleep(1./60)

        if feedback_timed_out is True:
            print('Failed to burn in policy due to lack of obs')
            sys.exit()

        # Clear memory
        self.dextrah_fgp.reset_hidden_state()

    def run(self):
        # Main control loop
        control_iter = 0
        print_iter = 60
        loop_time_filtered = 0.

        # Publish init position once
        self.publish_init_pos()
        time.sleep(1.)

        # Burn in
        print('Burning in')
        self.burn_in()

        print('Engaging policy')
        while rclpy.ok():

            # Set time start of loop
            start = time.time()

            # Pack and prepare observations
            state, left_image, right_image, feedback_timed_out = self.compute_fgp_observation()

            #img_dict = {'left_img': left_image,
            #            'right_img': right_image}
            #torch.save(img_dict, 'images.pth')

#            break

            # Query the FGP for actions
            # NOTE: publisher callbacks will pull action data from tensors
            # into lists themselves and publish
            # TODO: if times out again, probably should send robot to home position
            if not feedback_timed_out:
                self.compute_actions(state, left_image, right_image)
            else:
                print('not computingn actions')

            # Keep 60 Hz tick rate
            while (time.time() - start) < self._publish_dt:
                time.sleep(.00001)

            # Print control loop frequencies
            loop_time = time.time() - start
            alpha = 0.5
            if control_iter == 0:
                loop_time_filtered = loop_time
            else:
                loop_time_filtered = alpha * loop_time + (1. - alpha) * loop_time_filtered
            if (control_iter % print_iter) == 0:
                print('avg control rate', 1./loop_time_filtered)

            control_iter += 1

    def test_spinning(self):

        while rclpy.ok():
            start = time.time()
            # Keep 60 Hz tick rate
            while (time.time() - start) < self._publish_dt:
                time.sleep(.00001)

if __name__ == "__main__":
    print("Starting DextrAH FGP node")
    rclpy.init()

    # Create the fabric
    node_name = "dextrah_fgp_stereo"
    dextrah_fgp_node = DextrahFGPNode(node_name)

    # Spawn separate thread that spools the fabric
    spin_thread = Thread(target=rclpy.spin, args=(dextrah_fgp_node,), daemon=True)
    spin_thread.start()
    
    # Give time for data to flow
    time.sleep(1.)

    # Start the main dextrah loop
    dextrah_fgp_node.run()
    #dextrah_fgp_node.test_spinning()

    # Destroy node and shut down ROS
    dextrah_fgp_node.destroy_node()
    rclpy.shutdown()

    print('DextrAH FGP closed.')
