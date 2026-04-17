"""
openarm_policy_node.py — rl_games student 정책 ROS2 노드
──────────────────────────────────────────────────────────
eval.py 기반. a2c_mono_transformer (또는 stereo) student model 사용.

Subscribe:
  /joint_states                   sensor_msgs/JointState
  /head_camera/rgb/image_raw      sensor_msgs/Image   (H×W, rgb8/bgr8)
  /wrist_camera/rgb/image_raw     sensor_msgs/Image
  /goal_position                  geometry_msgs/Point  (선택, 없으면 --goal 사용)

Publish:
  /joint_position_targets         sensor_msgs/JointState
  /gripper_command                std_msgs/Float64

Usage:
  source /opt/ros/humble/setup.bash
  cd ~/isaac_ws/DEXTRAH_CAM/dextrah_lab
  python3 deployment_scripts/openarm_policy_node.py \
      --checkpoint distillation/runs/.../nn/student.pth \
      --student_cfg tasks/openarm/agents/rl_games_ppo_mono_transformer.yaml \
      --urdf ~/isaac_ws/openarm_bimanual.urdf \
      --goal 0.25 0.15 0.30
"""

import argparse
import pathlib
import sys
import threading

import numpy as np
import pinocchio as pin
import torch
import yaml
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from sensor_msgs.msg import JointState, Image
from geometry_msgs.msg import Point
from std_msgs.msg import Float64

# rl_games
from rl_games.algos_torch import model_builder
from rl_games.algos_torch.model_builder import ModelBuilder

# dextrah_lab student builders (eval.py와 동일)
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from distillation.a2c_mono_transformer import A2CBuilder as A2CMonoTransformerBuilder
from distillation.a2c_stereo_transformer import A2CBuilder as A2CStereoTransformerBuilder
from distillation.a2c_with_aux_cnn import A2CBuilder as A2CWithAuxCNNBuilder
from distillation.a2c_with_aux_cnn_stereo import A2CBuilder as A2CWithAuxCNNStereoBuilder
from distillation.a2c_mono_resnet import A2CBuilder as A2CMonoResnetBuilder

# ─── 로봇 설정 ────────────────────────────────────────────────────────────────
LEFT_ARM_JOINTS  = [f"openarm_left_joint{i}" for i in range(1, 8)]
LEFT_GRIPPER_JOINT = "openarm_left_finger_joint1"
LEFT_EE_FRAME    = "openarm_left_hand_tcp"

NUM_PROPRIO  = 38   # student_policy_obs 차원
NUM_ACTIONS  = 7    # 6 TCP twist + 1 gripper

ACTION_SCALE_LINEAR  = 0.02   # m per step
ACTION_SCALE_ANGULAR = 0.1    # rad per step


# ─── 유틸 ─────────────────────────────────────────────────────────────────────

def load_param_dict(cfg_path: str) -> dict:
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def adjust_state_dict_keys(ckpt_sd: dict, model_sd: dict) -> dict:
    """eval.py의 adjust_state_dict_keys와 동일."""
    out = {}
    for key, val in ckpt_sd.items():
        if key in model_sd:
            out[key] = val
        else:
            parts = key.split(".")
            parts.insert(2, "_orig_mod")
            new_key = ".".join(parts)
            if new_key in model_sd:
                out[new_key] = val
            else:
                no_orig = key.replace("_orig_mod.", "")
                out[no_orig if no_orig in model_sd else key] = val
    return out


def decode_rgb_full(msg: Image) -> np.ndarray | None:
    """sensor_msgs/Image → float32 (3, H, W) RGB [0, 1]."""
    if msg.encoding == "rgb8":
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3).astype(np.float32) / 255.0
    elif msg.encoding == "bgr8":
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3).astype(np.float32) / 255.0
        arr = arr[:, :, ::-1].copy()
    else:
        return None
    return arr.transpose(2, 0, 1)  # (3, H, W)


def resize_img(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    if arr.shape[-2:] == (h, w):
        return arr
    from PIL import Image as PILImage
    if arr.ndim == 2:
        img = PILImage.fromarray((arr * 255).astype(np.uint8)).resize((w, h), PILImage.BILINEAR)
        return np.array(img).astype(np.float32) / 255.0
    else:  # (3, H, W)
        chans = [PILImage.fromarray((arr[c] * 255).astype(np.uint8)).resize((w, h), PILImage.BILINEAR) for c in range(3)]
        return np.stack([np.array(c).astype(np.float32) / 255.0 for c in chans])


# ─── IK Solver ────────────────────────────────────────────────────────────────

class IKSolver:
    def __init__(self, model: pin.Model, ee_frame: str,
                 max_iter=200, eps=1e-4, damping=1e-6, dt=0.1):
        self.model   = model
        self.data    = model.createData()
        self.ee_id   = model.getFrameId(ee_frame)
        self.max_iter = max_iter
        self.eps      = eps
        self.damping  = damping
        self.dt       = dt

    def solve(self, q_init: np.ndarray, target: pin.SE3) -> np.ndarray:
        q = q_init.copy()
        for _ in range(self.max_iter):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            err = pin.log6(self.data.oMf[self.ee_id].actInv(target)).vector
            if np.linalg.norm(err) < self.eps:
                break
            J = pin.getFrameJacobian(self.model, self.data, self.ee_id, pin.LOCAL_WORLD_ALIGNED)
            dq = J.T @ np.linalg.solve(J @ J.T + self.damping * np.eye(6), err)
            q = pin.integrate(self.model, q, dq * self.dt)
        return q


# ─── ROS2 Node ────────────────────────────────────────────────────────────────

class OpenarmStudentNode(Node):
    def __init__(self, args):
        super().__init__("openarm_student_node")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args   = args
        self.stereo = args.stereo

        # ── student 모델 로드 (eval.py 방식) ──
        self._register_networks()
        network_params = load_param_dict(args.student_cfg)["params"]
        normalize_value = network_params["config"]["normalize_value"]
        normalize_input = network_params["config"]["normalize_input"]

        model_config = {
            "actions_num":    NUM_ACTIONS,
            "input_shape":    (NUM_PROPRIO,),
            "batch_size":     1,
            "num_seqs":       1,
            "value_size":     1,
            "normalize_value": normalize_value,
            "normalize_input": normalize_input,
        }
        builder = ModelBuilder()
        network = builder.load(network_params)
        self.model = network.build(model_config).to(self.device)
        self.model.eval()

        weights = torch.load(args.checkpoint, map_location=self.device)
        if "model" in weights:
            weights["model"] = adjust_state_dict_keys(weights["model"], self.model.state_dict())
            self.model.load_state_dict(weights["model"])
        else:
            self.model.load_state_dict(weights)
        if normalize_input and "running_mean_std" in weights:
            self.model.running_mean_std.load_state_dict(weights["running_mean_std"])

        self.get_logger().info(f"Student model loaded: {args.checkpoint}")

        # RNN 상태
        self.is_rnn = self.model.is_rnn()
        if self.is_rnn:
            self.hidden_states = tuple(s.to(self.device) for s in self.model.get_default_rnn_state())

        # prev_actions (rl_games 더미)
        self.prev_actions = torch.zeros((1, NUM_ACTIONS), device=self.device)

        # ── Pinocchio ──
        urdf = str(args.urdf).replace("~", str(pathlib.Path.home()))
        self.pin_model, _, _ = pin.buildModelsFromUrdf(urdf)
        self.pin_data  = self.pin_model.createData()
        self.ik_solver = IKSolver(self.pin_model, LEFT_EE_FRAME)
        self.get_logger().info(f"URDF loaded: {urdf}")

        # ── 상태 버퍼 ──
        self._lock         = threading.Lock()
        self.joint_pos     = np.zeros(7)
        self.joint_vel     = np.zeros(7)
        self.gripper_pos   = 0.0
        self.tcp_pos       = np.zeros(3)
        self.tcp_quat      = np.array([1., 0., 0., 0.])  # w,x,y,z
        self.tcp_vel       = np.zeros(6)
        self._prev_tcp_pos = np.zeros(3)
        self.goal          = np.array(args.goal, dtype=np.float32)
        self.prev_action   = np.zeros(NUM_ACTIONS, dtype=np.float32)

        # 이미지 크기 (args에서)
        self.img_h = args.img_h
        self.img_w = args.img_w
        if self.stereo:
            self.head_img  = np.zeros((3, self.img_h, self.img_w), dtype=np.float32)
            self.wrist_img = np.zeros((3, self.img_h, self.img_w), dtype=np.float32)
        else:
            self.head_img  = np.zeros((3, self.img_h, self.img_w), dtype=np.float32)
            self.wrist_img = np.zeros((3, self.img_h, self.img_w), dtype=np.float32)

        # ── QoS ──
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=1,
        )

        # ── Subscribers ──
        self.create_subscription(JointState, "/joint_states",            self._joint_cb,       qos)
        self.create_subscription(Image,      "/head_camera/rgb/image_raw",  self._head_cb,     qos)
        self.create_subscription(Image,      "/wrist_camera/rgb/image_raw", self._wrist_cb,    qos)
        self.create_subscription(Point,      "/goal_position",            self._goal_cb,        10)

        # ── Publishers ──
        self.pub_joints  = self.create_publisher(JointState, "/joint_position_targets", 10)
        self.pub_gripper = self.create_publisher(Float64,    "/gripper_command",         10)

        # ── 추론 타이머 60 Hz ──
        self.create_timer(1.0 / 60.0, self._step)
        self.get_logger().info("OpenArm Student Node started @ 60 Hz")

    def _register_networks(self):
        model_builder.register_network("a2c_mono_transformer",    A2CMonoTransformerBuilder)
        model_builder.register_network("a2c_stereo_transformer",  A2CStereoTransformerBuilder)
        model_builder.register_network("a2c_aux_cnn_net",         A2CWithAuxCNNBuilder)
        model_builder.register_network("a2c_aux_cnn_net_stereo",  A2CWithAuxCNNStereoBuilder)
        model_builder.register_network("a2c_mono_resnet",         A2CMonoResnetBuilder)

    # ── Callbacks ──────────────────────────────────────────────────────────────

    def _joint_cb(self, msg: JointState):
        name_idx = {n: i for i, n in enumerate(msg.name)}
        with self._lock:
            for j, jname in enumerate(LEFT_ARM_JOINTS):
                if jname in name_idx:
                    i = name_idx[jname]
                    self.joint_pos[j] = msg.position[i]
                    self.joint_vel[j] = msg.velocity[i] if msg.velocity else 0.0
            if LEFT_GRIPPER_JOINT in name_idx:
                self.gripper_pos = msg.position[name_idx[LEFT_GRIPPER_JOINT]]

            # FK → TCP pose
            q = np.zeros(self.pin_model.nq)
            q[:7] = self.joint_pos
            pin.forwardKinematics(self.pin_model, self.pin_data, q)
            pin.updateFramePlacements(self.pin_model, self.pin_data)
            oMf = self.pin_data.oMf[self.pin_model.getFrameId(LEFT_EE_FRAME)]
            self.tcp_pos  = oMf.translation.copy()
            xyzw = pin.Quaternion(oMf.rotation).coeffs()
            self.tcp_quat = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])

            dt = 1.0 / 60.0
            self.tcp_vel[:3] = (self.tcp_pos - self._prev_tcp_pos) / dt
            self._prev_tcp_pos = self.tcp_pos.copy()

    def _head_cb(self, msg: Image):
        arr = decode_rgb_full(msg)
        if arr is not None:
            with self._lock:
                self.head_img = resize_img(arr, self.img_h, self.img_w)

    def _wrist_cb(self, msg: Image):
        arr = decode_rgb_full(msg)
        if arr is not None:
            with self._lock:
                self.wrist_img = resize_img(arr, self.img_h, self.img_w)

    def _goal_cb(self, msg: Point):
        with self._lock:
            #self.goal = np.array([msg.x, msg.y, msg.z], dtype=np.float32)
            self.goal = np.array([0.25, 0.15, 0.29], dtype=np.float32)

    # ── 추론 ───────────────────────────────────────────────────────────────────

    def _step(self):
        with self._lock:
            joint_pos  = self.joint_pos.copy()
            joint_vel  = self.joint_vel.copy()
            gripper    = self.gripper_pos
            tcp_pos    = self.tcp_pos.copy()
            tcp_quat   = self.tcp_quat.copy()
            tcp_vel    = self.tcp_vel.copy()
            goal       = self.goal.copy()
            prev_act   = self.prev_action.copy()
            head_img   = self.head_img.copy()
            wrist_img  = self.wrist_img.copy()

        # ── proprio (38-dim, eval.py의 student_policy_obs와 동일) ──
        proprio = np.concatenate([
            joint_pos,   # 7  robot_dof_pos_noisy
            joint_vel,   # 7  robot_dof_vel_noisy
            [gripper],   # 1  left_gripper_joint_pos
            tcp_pos,     # 3  left_tcp_pose[:3]
            tcp_quat,    # 4  body_pose_w quat (w,x,y,z)
            tcp_vel,     # 6  left_tcp_vel
            goal,        # 3  object_goal
            prev_act,    # 7  actions
        ]).astype(np.float32)  # = 38

        obs_t     = torch.from_numpy(proprio).float().unsqueeze(0).to(self.device)
        head_t    = torch.from_numpy(head_img).float().unsqueeze(0).to(self.device)   # (1,3,H,W)
        wrist_t   = torch.from_numpy(wrist_img).float().unsqueeze(0).to(self.device)  # (1,3,H,W)

        # ── batch_dict (eval.py의 get_actions와 동일) ──
        batch_dict = {
            "is_train":    False,
            "obs":         obs_t,
            "prev_actions": self.prev_actions,
        }
        if self.stereo:
            batch_dict["img_left"]  = head_t
            batch_dict["img_right"] = wrist_t
        else:
            batch_dict["img"]     = head_t   # mono: head만 사용
            batch_dict["rgb"]     = head_t
            batch_dict["rgb_data"] = head_t

        if self.is_rnn:
            batch_dict["rnn_states"]  = self.hidden_states
            batch_dict["seq_length"]  = 1
            batch_dict["rnn_masks"]   = None

        batch_dict["finetune_backbone"] = False

        with torch.no_grad():
            res_dict = self.model(batch_dict)

        if self.is_rnn:
            rnn_out = res_dict["rnn_states"]
            self.hidden_states = tuple(
                s.detach() for s in (rnn_out[0] if isinstance(rnn_out[0], tuple) else rnn_out)
            )

        # deterministic action (eval.py와 동일, mus 사용)
        action = res_dict["mus"].squeeze(0).cpu().numpy()
        action = np.clip(action, -1.0, 1.0)

        self.prev_actions = torch.from_numpy(action).float().unsqueeze(0).to(self.device)
        with self._lock:
            self.prev_action = action.copy()

        # ── 액션 해석 ──
        delta_pos = action[:3] * ACTION_SCALE_LINEAR
        delta_rpy = action[3:6] * ACTION_SCALE_ANGULAR
        gripper_cmd = 0.044 if action[6] > 0.0 else 0.0

        target_pos = tcp_pos + delta_pos
        curr_rpy   = R.from_quat([tcp_quat[1], tcp_quat[2], tcp_quat[3], tcp_quat[0]]).as_euler("xyz")
        target_rot = R.from_euler("xyz", curr_rpy + delta_rpy).as_matrix()
        target_se3 = pin.SE3(target_rot, target_pos)

        # ── IK ──
        q_init = np.zeros(self.pin_model.nq)
        q_init[:7] = joint_pos
        q_sol = self.ik_solver.solve(q_init, target_se3)

        # ── Publish ──
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name     = LEFT_ARM_JOINTS
        js.position = q_sol[:7].tolist()
        self.pub_joints.publish(js)

        gm = Float64()
        gm.data = gripper_cmd
        self.pub_gripper.publish(gm)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",   required=True, help="student 체크포인트 .pth")
    parser.add_argument("--student_cfg",  required=True, help="student yaml 설정 경로")
    parser.add_argument("--urdf",         default="~/isaac_ws/openarm_bimanual.urdf")
    parser.add_argument("--goal",         nargs=3, type=float, default=[0.25, 0.15, 0.30],
                        metavar=("X","Y","Z"))
    parser.add_argument("--img_h",        type=int, default=192)
    parser.add_argument("--img_w",        type=int, default=240)
    parser.add_argument("--stereo",       action="store_true", default=True,
                        help="스테레오 모델 사용 (기본: mono)")
    args = parser.parse_args()

    rclpy.init()
    node = OpenarmStudentNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
