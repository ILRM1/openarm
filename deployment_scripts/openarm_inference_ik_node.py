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
from geometry_msgs.msg import Vector3Stamped
from sensor_msgs.msg import JointState, Image
from tf2_ros import TransformBroadcaster
from std_msgs.msg import Bool

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

URDF_PATH       = "/home/neubility-sim/isaac_ws/openarm_description/urdf/robot/openarm_bimanual.urdf"
#URDF_PATH       = "/home/neubility-sim/openarm_ros2_ws/src/openarm_description/urdf/robot/openarm_bimanual.urdf"
EE_FRAME        = "openarm_left_hand_tcp"
JOINT_NAMES     = [f"openarm_left_joint{i}" for i in range(1, 8)]
LAMBDA_VAL = 0.01
POSITION_ONLY   = False

JOINT_STATE_TOPIC  = "/joint_states"
TARGET_POSE_TOPIC  = "/left/ik_target_pose"
PUBLISH_TOPIC      = "/arm/command"

# ─────────────────────────────────────────
# IK solver 
# ─────────────────────────────────────────
@torch.jit.script
def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    shape = q.shape
    q = q.reshape(-1, 4)
    return torch.cat((q[..., 0:1], -q[..., 1:]), dim=-1).view(shape)

@torch.jit.script
def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    if q1.shape != q2.shape:
        msg = f"Expected input quaternion shape mismatch: {q1.shape} != {q2.shape}."
        raise ValueError(msg)
    # reshape to (N, 4) for multiplication
    shape = q1.shape
    q1 = q1.reshape(-1, 4)
    q2 = q2.reshape(-1, 4)
    # extract components from quaternions
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    # perform multiplication
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    return torch.stack([w, x, y, z], dim=-1).view(shape)

@torch.jit.script
def axis_angle_from_quat(quat: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    quat = quat * (1.0 - 2.0 * (quat[..., 0:1] < 0.0))
    mag = torch.linalg.norm(quat[..., 1:], dim=-1)
    half_angle = torch.atan2(mag, quat[..., 0])
    angle = 2.0 * half_angle
    # check whether to apply Taylor approximation
    sin_half_angles_over_angles = torch.where(
        angle.abs() > eps, torch.sin(half_angle) / angle, 0.5 - angle * angle / 48
    )
    return quat[..., 1:4] / sin_half_angles_over_angles.unsqueeze(-1)

def compute_pose_error(
    t01: torch.Tensor,
    q01: torch.Tensor,
    t02: torch.Tensor,
    q02: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    
    source_quat_norm = quat_mul(q01, quat_conjugate(q01))[:, 0]
    # q_current_inv = q_current_conj / q_current_norm
    source_quat_inv = quat_conjugate(q01) / source_quat_norm.unsqueeze(-1)
    # q_error = q_target * q_current_inv
    quat_error = quat_mul(q02, source_quat_inv)

    # Compute position error
    pos_error = t02 - t01

    axis_angle_error = axis_angle_from_quat(quat_error)
    return pos_error, axis_angle_error


def _compute_delta_joint_pos(delta_pose: torch.Tensor, jacobian: torch.Tensor) -> torch.Tensor:
    # parameters
    lambda_val = LAMBDA_VAL
    # computation
    jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)
    lambda_matrix = (lambda_val**2) * torch.eye(n=jacobian.shape[1], device=delta_pose.device)
    delta_joint_pos = (
        jacobian_T @ torch.inverse(jacobian @ jacobian_T + lambda_matrix) @ delta_pose.unsqueeze(-1)
    )
    delta_joint_pos = delta_joint_pos.squeeze(-1)

    return delta_joint_pos


def compute(ee_pos: torch.Tensor, ee_quat: torch.Tensor, 
            ee_pos_des: torch.Tensor, ee_quat_des: torch.Tensor,
            jacobian: torch.Tensor, joint_pos: torch.Tensor) -> torch.Tensor:
    
    position_error, axis_angle_error = compute_pose_error(ee_pos, ee_quat, ee_pos_des, ee_quat_des)
    pose_error = torch.cat((position_error, axis_angle_error), dim=1)
    delta_joint_pos = _compute_delta_joint_pos(delta_pose=pose_error, jacobian=jacobian)

    return joint_pos + delta_joint_pos


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

        self.latest_joint_state: Optional[JointState] = None
        self.name_to_idx = {}

        # Set up cuda and warp
        self.device='cuda'

        with open(URDF_PATH) as f:
            urdf_str = f.read()
        self.chain = pk.build_serial_chain_from_urdf(urdf_str, EE_FRAME).to(dtype=torch.float32, device=self.device)
        self.batch_size = 1
        self.num_actions = 7
        self.num_obs = 24
        self.action_scale = (0.02, 0.1, 0.044)
        self.tcp_pose_min = torch.tensor([0.05, 0., 0.21, -np.pi*2, -np.pi*2, -np.pi*2], device=self.device)
        self.tcp_pose_max = torch.tensor([0.45, 0.4, 0.35, np.pi*2, np.pi*2, np.pi*2], device=self.device)
        self.student_ckpt = "distillation/runs/dextrah_student_20000_iters.pth"

        # Image saving
        self._save_images = True
        self._save_dir = os.path.join(os.path.dirname(__file__), "saved_images")
        self._save_interval = 30  # save every N control iterations
        os.makedirs(self._save_dir, exist_ok=True)

        # For converting ROS image messages to CV formates
        self.bridge = CvBridge()

        # Camera subscriber
        self._left_image_lock = Lock()
        self._right_image_lock = Lock()
        self._left_image = None
        self._right_image = None
        self._image_height = 192
        self._image_width = 240

        # NOTE: expecting 1/2 the resolution
        self._downsample_factor = 1
        self.camera_left_feedback_time = time.time()
        self.camera_left_sub = self.create_subscription(Image, '/camera/image', self._left_camera_callback, 10)

        self.camera_right_feedback_time = time.time()
        self.camera_right_sub = self.create_subscription(Image, '/wristcam_left/image',self._right_camera_callback,10)

        # Robot subscribers
        self.robot_q = torch.zeros(self.batch_size, 16, device=self.device)
        self.robot_qd = torch.zeros(self.batch_size, 16, device=self.device)
        self.left_tcp_pose = torch.zeros(self.batch_size, 6, device=self.device)
        self.openarm_feedback_time = time.time()
        self.left_tcp_pose_feedback_time = time.time()
        self._openarm_joint_position_lock = Lock()
        self._left_tcp_pose_sub_lock = Lock()
        self._openarm_sub = self.create_subscription(JointState(), '/joint_states', self._openarm_sub_callback, 1)
        self._tcp_pose_sub = self.create_subscription(PoseStamped(), '/tcp_pose/left', self._tcp_pose_sub_callback, 1)

        self._publish_rate = 60. # Hz
        self._publish_dt = 1./self._publish_rate # s

        # Goal to bring object to. NOTE: this should be a command in the future
        self.object_goal = torch.tensor([0.25, 0.15, 0.29], device=self.device).repeat((self.batch_size, 1))

        self.last_actions = torch.zeros(self.batch_size, self.num_actions, device=self.device)

        # publishers
        # For commanding pose target
        self.left_tcp_pose_targets = None
        self._left_tcp_pose_lock = Lock()
        self._left_tcp_pose_command_pub = self.create_publisher(PoseStamped, '/left/ik_target_pose', 1)
        self._left_tcp_pose_timer = self.create_timer(self._publish_dt, self._left_tcp_pose_pub_callback)

        self.left_gripper_pos_targets = None
        self._left_gripper_pos_lock = Lock()

        self.arm_pos_targets = None
        self._arm_pos_lock = Lock()
        self.arm_pos_pub = self.create_publisher(JointState, '/arm/command', 1)
        self._arm_pos_timer = self.create_timer(self._publish_dt, self.publish_arm_cmd)

        # Create publisher for predicted object pose
        self._object_pos_lock = Lock()
        self.object_pos = None
#        self._object_pos_pub = self.create_publisher(
#            PoseStamped,
#            "/kuka_allegro_fabric/predicted_obj_pose",
#            1)
        self.object_pos_tf = TransformBroadcaster(self)
        self._fgp_object_pos_timer = self.create_timer(self._publish_dt,
                                                       self._object_pos_callback)

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

    def _left_camera_callback(self, msg):
        '''
        TODO: UPDATE
        '''
        # Convert ROS camera data to np array of shape height, width, channel
        #img_np = np.frombuffer(
        #    msg.data, dtype=np.uint8).reshape((self._image_height, self._image_width, 3)).astype(np.float32)

        # Convert ROS image to numpy image with h x w x 3, and rgb ordering
        img_np = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8').astype(np.float32)

        # Interpolate down to the target image size
        img_np = cv2.resize(
            img_np,
            (self._image_width//self._downsample_factor, self._image_height//self._downsample_factor),
            interpolation=cv2.INTER_LINEAR
        )

#        bgr_image = cv2.cvtColor(img_np.astype(np.uint8), cv2.COLOR_RGB2BGR)
#        cv2.imshow('Left RGB Image', bgr_image)
#
#        cv2.waitKey(1)

        # Reshape into (3, height, width)
        img_np = np.transpose(img_np, (2, 0, 1))

        # Scale to be between [0, 1]
        img_np /= 255.

        # Move to torch tensor and make sure shape is 1x3xhxw
        with self._left_image_lock:
            self.camera_left_feedback_time = time.time()
            self._left_image = torch.from_numpy(img_np).to(self.device).unsqueeze(0)

    def _right_camera_callback(self, msg):
        '''
        TODO: UPDATE
        '''
        # Convert ROS camera data to np array of shape height, width, channel
        #img_np = np.frombuffer(
        #    msg.data, dtype=np.uint8).reshape((self._image_height, self._image_width, 3)).astype(np.float32)

        # Convert ROS image to numpy image with h x w x 3, and rgb ordering
        img_np = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8').astype(np.float32)

        # Interpolate down to the target image size
        img_np = cv2.resize(
            img_np,
            (self._image_width//self._downsample_factor, self._image_height//self._downsample_factor),
            interpolation=cv2.INTER_LINEAR
        )

        # bgr_image = cv2.cvtColor(img_np.astype(np.uint8), cv2.COLOR_RGB2BGR)
        # cv2.imshow('Right RGB Image', bgr_image)

        # cv2.waitKey(1)

        # Reshape into (3, height, width)
        img_np = np.transpose(img_np, (2, 0, 1))

        # Scale to be between [0, 1]
        img_np /= 255.

        # Move to torch tensor and make sure shape is 1x3xhxw
        with self._right_image_lock:
            self.camera_right_feedback_time = time.time()
            self._right_image = torch.from_numpy(img_np).to(self.device).unsqueeze(0)

    def _openarm_sub_callback(self, msg):

        name_idx = {n: i for i, n in enumerate(msg.name)}

        left_joints  = [f"openarm_left_joint{i}"  for i in range(1, 8)]
        right_joints = [f"openarm_right_joint{i}" for i in range(1, 8)]

        def extract(joints, field):
            arr = getattr(msg, field)
            return [arr[name_idx[j]] if (j in name_idx and arr) else 0.0 for j in joints]

        l_pos = extract(left_joints,  "position")
        l_vel = extract(left_joints,  "velocity")
        r_pos = extract(right_joints, "position")
        r_vel = extract(right_joints, "velocity")

        lf_pos = extract(["openarm_left_finger_joint1"],  "position")
        lf_vel = extract(["openarm_left_finger_joint1"],  "velocity")
        rf_pos = extract(["openarm_right_finger_joint1"], "position")
        rf_vel = extract(["openarm_right_finger_joint1"], "velocity")

        with self._openarm_joint_position_lock:
            self.openarm_feedback_time = time.time()
            self.robot_q[:,  :7].copy_(torch.tensor([l_pos],  dtype=torch.float32, device=self.device))
            self.robot_qd[:, :7].copy_(torch.tensor([l_vel],  dtype=torch.float32, device=self.device))
            self.robot_q[:,  7:14].copy_(torch.tensor([r_pos], dtype=torch.float32, device=self.device))
            self.robot_qd[:, 7:14].copy_(torch.tensor([r_vel], dtype=torch.float32, device=self.device))
            self.robot_q[:,  14:15].copy_(torch.tensor([lf_pos], dtype=torch.float32, device=self.device))
            self.robot_qd[:, 14:15].copy_(torch.tensor([lf_vel], dtype=torch.float32, device=self.device))
            self.robot_q[:,  15:16].copy_(torch.tensor([rf_pos], dtype=torch.float32, device=self.device))
            self.robot_qd[:, 15:16].copy_(torch.tensor([rf_vel], dtype=torch.float32, device=self.device))

            self.latest_joint_state = msg
            self.name_to_idx = name_idx

    def _tcp_pose_sub_callback(self, msg):

        p = msg.pose.position
        o = msg.pose.orientation
        left_tcp_pos = torch.tensor(
            [p.x, p.y, p.z], dtype=torch.float32, device=self.device
        )

        quat_xyzw = [o.x, o.y, o.z, o.w]
   
        euler = R.from_quat(quat_xyzw).as_euler('xyz')
        euler = torch.tensor(euler, dtype=torch.float32, device=self.device)

        left_tcp_pose = torch.cat((left_tcp_pos, euler), dim=-1).to(dtype=left_tcp_pos.dtype)
       
        # Copy over to robot state tensors
        with self._left_tcp_pose_sub_lock:
            self.left_tcp_pose_feedback_time = time.time()
            self.left_tcp_pose.copy_(left_tcp_pose)

    def _left_tcp_pose_pub_callback(self):
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


    def publish_arm_cmd(self):
        arm_pos_targets = None
        with self._arm_pos_lock:
            if self.arm_pos_targets is not None:
                arm_pos_targets = self.arm_pos_targets[0,:].float().detach().cpu().numpy()
            if self.left_gripper_pos_targets is not None:
                left_gripper_pos_targets = self.left_gripper_pos_targets.float().detach().cpu().numpy()

        if arm_pos_targets is not None:
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = [
                'openarm_left_joint1',  
                'openarm_left_joint2',  
                'openarm_left_joint3',  
                'openarm_left_joint4',  
                'openarm_left_joint5',  
                'openarm_left_joint6',  
                'openarm_left_joint7',  
                'openarm_left_finger_joint1', 
            ]
            msg.position = arm_pos_targets.tolist() + left_gripper_pos_targets.tolist()
            msg.velocity = []
            msg.effort = []
            self.arm_pos_pub.publish(msg)

    def publish_init_pos(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [
            'openarm_left_joint1',  
            'openarm_left_joint2',  
            'openarm_left_joint3',  
            'openarm_left_joint4',  
            'openarm_left_joint5',  
            'openarm_left_joint6',  
            'openarm_left_joint7',  
            'openarm_left_finger_joint1', 

            'openarm_right_joint1',
            'openarm_right_joint2',
            'openarm_right_joint3',
            'openarm_right_joint4',
            'openarm_right_joint5',
            'openarm_right_joint6',
            'openarm_right_joint7',
            'openarm_right_finger_joint1',
        ]
        msg.position = [0.9, -0.35, -0.24, 2.0, -0.54, 0., 1.1, 0.044, 
                        -0.9, 0.35, 0.24, 2.0, 0.54, 0., -1.1, 0.044]
        msg.velocity = []
        msg.effort = []
        self.arm_pos_pub.publish(msg)
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
            # engage_fgp = False
            # with self._engage_fgp_lock:
            #     engage_fgp = self.engage_fgp
            engage_fgp = True
            if engage_fgp:
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


    def compute_observation(self):

        feedback_timed_out = False

        with self._left_tcp_pose_sub_lock:
            end = time.time()

            left_tcp_pose = self.left_tcp_pose.clone()

            if (end - self.left_tcp_pose_feedback_time) > (3. * self._publish_dt):
                print('no feedback from left tcp pose')
                feedback_timed_out = True
        

        with self._openarm_joint_position_lock:
            end = time.time()

            robot_q = self.robot_q[:, :7].clone()
            gipper_q = self.robot_q[:, 14].unsqueeze(-1).clone()

            if (end - self.openarm_feedback_time) > (3. * self._publish_dt):
                print('no feedback from openarm joints')
                feedback_timed_out = True

        state = torch.cat(
            (
                robot_q,
                gipper_q,
                left_tcp_pose,
                self.object_goal, 
                self.last_actions,
            ),
            dim=-1
        )

        # Copy camera image
        left_image = None
        with self._left_image_lock:
            end = time.time()
            if (end - self.camera_left_feedback_time) > (3. * self._publish_dt * 2): 
                print('no feedback from left camera')
                feedback_timed_out = True
            left_image = torch.clone(self._left_image)

        right_image = None
        with self._right_image_lock:
            end = time.time()
            if (end - self.camera_right_feedback_time) > (3. * self._publish_dt * 2): 
                print('no feedback from right camera')
                feedback_timed_out = True
            right_image = torch.clone(self._right_image)

        return state, left_image, right_image, feedback_timed_out


    def compute_actions(self, state, left_image, right_image, transmit=True, save_pth=False):
#        with torch.no_grad():
#            action_dict = self.dextrah_fgp.step(state, left_image, right_image)

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

        left_tcp_pose = state[:, 8:14].clone()

        left_tcp_pos_targets = left_tcp_pose[:,:3] + actions[:,:3] * self.action_scale[0]
        left_tcp_ori_targets = left_tcp_pose[:,3:6] + actions[:,3:6] * self.action_scale[1]
        
        left_tcp_pose_targets = torch.cat((left_tcp_pos_targets, left_tcp_ori_targets), dim=-1)
        left_tcp_pose_targets = torch.max(torch.min(left_tcp_pose_targets, self.tcp_pose_max), self.tcp_pose_min)
       
        # euler → quaternion (w,x,y,z)
        euler_np = left_tcp_pose_targets[0, 3:6].cpu().detach().numpy()
        euler_np[0]=euler_np[0]*-1.
        euler_np[1]=euler_np[1]*1.
        euler_np[2]=euler_np[2]*-1.
        tq = R.from_euler('xyz', euler_np).as_quat() 
        target_quat = torch.tensor([[tq[3], tq[0], tq[1], tq[2]]], dtype=torch.float32, device=self.device)
        left_tcp_pose_targets = torch.cat((left_tcp_pose_targets[:, :3], target_quat), dim=-1)
     
        left_gripper_action = 0.5 * (actions[:,6].clone() + 1.)
        left_gripper_pos_target = torch.where(left_gripper_action>0.5, self.action_scale[2], 0.)

    
        #-------------------- Compute IK -------------------------
        js  = self.latest_joint_state
        n2i = self.name_to_idx

        # Build joint angle vector
        q_ctrl = np.array([js.position[n2i[name]] for name in JOINT_NAMES], dtype=np.float32)
        th = torch.tensor(q_ctrl, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, 7)

        # Forward kinematics + Jacobian
        T = self.chain.forward_kinematics(th).get_matrix()  # (1, 4, 4)
        ee_pos  = T[0, :3, 3].cpu().numpy()
        q_xyzw  = R.from_matrix(T[0, :3, :3].cpu().numpy()).as_quat()
        ee_quat = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])  # wxyz
      
        J = self.chain.jacobian(th)[0].cpu().numpy()  # (6, 7)

        #left_tcp_pose_targets[:,:3] = torch.tensor([[0.3, 0.05, 0.29]], dtype=torch.float32, device=self.device)
        # Target
        tgt_pos  = left_tcp_pose_targets[:,:3].cpu().detach().numpy().squeeze(0)
        tgt_quat = left_tcp_pose_targets[:,3:7].cpu().detach().numpy().squeeze(0)

        J_use   = J if not POSITION_ONLY else J[:3]

        q_cmd   = compute(ee_pos=torch.from_numpy(ee_pos).unsqueeze(0).float(),
                            ee_quat=torch.from_numpy(ee_quat).unsqueeze(0).float(),
                            ee_pos_des=torch.from_numpy(tgt_pos).unsqueeze(0).float(),
                            ee_quat_des=torch.from_numpy(tgt_quat).unsqueeze(0).float(),
                            jacobian=torch.from_numpy(J_use).unsqueeze(0).float(),
                            joint_pos=torch.from_numpy(q_ctrl).unsqueeze(0).float(),)
       

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

            with self._arm_pos_lock: 
                if self.arm_pos_targets is None:
                    self.arm_pos_targets = q_cmd
                else:
                    # In-place update to hand PCA targets
                    self.arm_pos_targets.copy_(q_cmd)          

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
            state, left_image, right_image, feedback_timed_out = self.compute_observation()

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
            state, left_image, right_image, feedback_timed_out = self.compute_observation()

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

