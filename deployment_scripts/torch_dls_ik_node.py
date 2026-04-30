#!/usr/bin/env python3
import math
from threading import Lock
from typing import Optional

import numpy as np
import pytorch_kinematics as pk
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from scipy.spatial.transform import Rotation as R
import torch
# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────
URDF_PATH       = "/home/neubility-sim/isaac_ws/openarm_description/urdf/robot/openarm_bimanual.urdf"
EE_FRAME        = "openarm_left_hand_tcp"
JOINT_NAMES     = [f"openarm_left_joint{i}" for i in range(1, 8)]
LAMBDA_VAL = 0.01
POSITION_ONLY   = False
CONTROL_RATE    = 60.0      # Hz

JOINT_STATE_TOPIC   = "/joint_states"
TARGET_POSE_TOPIC   = "/left/ik_target_pose"
GRIPPER_TOPIC       = "/left/gripper_command"
PUBLISH_TOPIC       = "/arm/command"
GRIPPER_JOINT_NAME  = "openarm_left_finger_joint1"

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


# ─────────────────────────────────────────
# ROS2 node
# ─────────────────────────────────────────
class DifferentialIKNode(Node):
    def __init__(self):
        super().__init__("differential_ik_node")
       
        # pytorch_kinematics chain
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        with open(URDF_PATH) as f:
            urdf_str = f.read()
        self.chain = pk.build_serial_chain_from_urdf(urdf_str, EE_FRAME).to(dtype=torch.float32, device=self.device)

        # State
        self._lock = Lock()
        self.latest_joint_state: Optional[JointState] = None
        self.latest_target_pose: Optional[PoseStamped] = None
        self.latest_gripper_pos: float = 0.0
        self.name_to_idx = {}

        # ROS2
        self.create_subscription(JointState, JOINT_STATE_TOPIC, self._joint_cb, 10)
        self.create_subscription(PoseStamped, TARGET_POSE_TOPIC, self._target_cb, 10)
        self.create_subscription(JointState, GRIPPER_TOPIC, self._gripper_cb, 10)
        self.pub = self.create_publisher(JointState, PUBLISH_TOPIC, 10)
        self.create_timer(1.0 / CONTROL_RATE, self._control_loop)

        self.get_logger().info(f"DifferentialIK started | EE: {EE_FRAME} | joints: {JOINT_NAMES}")

    def _joint_cb(self, msg: JointState):
        with self._lock:
            self.latest_joint_state = msg
            self.name_to_idx = {n: i for i, n in enumerate(msg.name)}

    def _target_cb(self, msg: PoseStamped):
        with self._lock:
            self.latest_target_pose = msg

    def _gripper_cb(self, msg: JointState):
        with self._lock:
            if msg.position:
                self.latest_gripper_pos = msg.position[0]

    def _control_loop(self):
        with self._lock:
            js  = self.latest_joint_state
            tgt = self.latest_target_pose
            n2i = self.name_to_idx

        if js is None or tgt is None:
            return

        # Forward kinematics + Jacobian (pytorch_kinematics)
        q_ctrl = np.array([js.position[n2i[name]] for name in JOINT_NAMES], dtype=np.float32)
        th = torch.tensor(q_ctrl, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, 7)

        T = self.chain.forward_kinematics(th).get_matrix()  # (1, 4, 4)
        ee_pos_t = T[0, :3, 3].unsqueeze(0)                 # (1, 3)
        q_xyzw = R.from_matrix(T[0, :3, :3].cpu().numpy()).as_quat()
        ee_quat_t = torch.tensor(
            [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=torch.float32, device=self.device
        ).unsqueeze(0)                                       # (1, 4) wxyz

        J = self.chain.jacobian(th)                          # (1, 6, 7)

        # Target
        p = tgt.pose
        tgt_pos_t  = torch.tensor(
            [p.position.x, p.position.y, p.position.z], dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        tgt_quat_t = torch.tensor(
            [p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z],
            dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        J_use = J if not POSITION_ONLY else J[:, :3, :]

        q_cmd = compute(
            ee_pos=ee_pos_t,
            ee_quat=ee_quat_t,
            ee_pos_des=tgt_pos_t,
            ee_quat_des=tgt_quat_t,
            jacobian=J_use,
            joint_pos=th,
        ).cpu().numpy().squeeze(0)
      
        # Publish arm + gripper
        with self._lock:
            gripper_pos = self.latest_gripper_pos
        out = JointState()
        out.header.stamp = self.get_clock().now().to_msg()
        out.name     = JOINT_NAMES + [GRIPPER_JOINT_NAME]
        out.position = q_cmd.tolist() + [gripper_pos]
        self.pub.publish(out)


def main():
    rclpy.init()
    node = DifferentialIKNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
