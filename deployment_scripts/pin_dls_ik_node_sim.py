#!/usr/bin/env python3
from threading import Lock
from typing import Optional

import numpy as np
import pinocchio as pin
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────
URDF_PATH           = "/home/neubility-sim/isaac_ws/openarm_description/urdf/robot/openarm_bimanual.urdf"
EE_FRAME            = "openarm_left_hand_tcp"
JOINT_NAMES         = [f"openarm_left_joint{i}" for i in range(1, 8)]
LAMBDA_VAL          = 0.01
POSITION_ONLY       = False
CONTROL_RATE        = 10.0      # Hz

JOINT_STATE_TOPIC   = "/joint_states"
TARGET_POSE_TOPIC   = "/left/ik_target_pose"
GRIPPER_TOPIC       = "/left/gripper_command"
PUBLISH_TOPIC       = "/arm/command"
GRIPPER_JOINT_NAME  = "openarm_left_finger_joint1"

# ─────────────────────────────────────────
# IK solver (numpy)
# ─────────────────────────────────────────
def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """q = [w, x, y, z]"""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product, q = [w, x, y, z]"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def axis_angle_from_quat(q: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """q = [w, x, y, z] → axis-angle (3,)"""
    if q[0] < 0:
        q = -q
    mag = np.linalg.norm(q[1:])
    half_angle = np.arctan2(mag, q[0])
    angle = 2.0 * half_angle
    s = np.sin(half_angle) / angle if abs(angle) > eps else 0.5 - angle * angle / 48.0
    return q[1:4] / s


def compute_pose_error(
    pos_cur: np.ndarray, quat_cur: np.ndarray,
    pos_des: np.ndarray, quat_des: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    norm_sq = np.dot(quat_cur, quat_cur)
    quat_inv = quat_conjugate(quat_cur) / norm_sq
    quat_err = quat_mul(quat_des, quat_inv)
    pos_err = pos_des - pos_cur
    aa_err = axis_angle_from_quat(quat_err)
    return pos_err, aa_err


def compute_dls(pose_error: np.ndarray, jacobian: np.ndarray) -> np.ndarray:
    """Δq = J^T (J J^T + λ²I)^{-1} Δx"""
    JT = jacobian.T
    lam2_I = (LAMBDA_VAL ** 2) * np.eye(jacobian.shape[0])
    return JT @ np.linalg.solve(jacobian @ JT + lam2_I, pose_error)


# ─────────────────────────────────────────
# ROS2 node
# ─────────────────────────────────────────
class DifferentialIKNode(Node):
    def __init__(self):
        super().__init__("differential_ik_node")

        # Pinocchio model
        self.model = pin.buildModelFromUrdf(URDF_PATH)
        self.data = self.model.createData()
        self.frame_id = self.model.getFrameId(EE_FRAME)

        # Map JOINT_NAMES → velocity/configuration indices in full model
        self.q_indices = []
        self.v_indices = []
        for name in JOINT_NAMES:
            jid = self.model.getJointId(name)
            jnt = self.model.joints[jid]
            self.q_indices.extend(range(jnt.idx_q, jnt.idx_q + jnt.nq))
            self.v_indices.extend(range(jnt.idx_v, jnt.idx_v + jnt.nv))

        # State
        self._lock = Lock()
        self.latest_joint_state: Optional[JointState] = None
        self.latest_target_pose: Optional[PoseStamped] = None
        self.latest_gripper_pos: float = 0.0
        self.name_to_idx: dict = {}

        # ROS2
        self.create_subscription(JointState, JOINT_STATE_TOPIC, self._joint_cb, 10)
        self.create_subscription(PoseStamped, TARGET_POSE_TOPIC, self._target_cb, 10)
        self.create_subscription(JointState, GRIPPER_TOPIC, self._gripper_cb, 10)
        self.pub = self.create_publisher(JointState, PUBLISH_TOPIC, 10)
        self.create_timer(1.0 / CONTROL_RATE, self._control_loop)

        self.get_logger().info(
            f"DifferentialIK (pinocchio) started | EE: {EE_FRAME} | joints: {JOINT_NAMES}"
        )

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

        # Current joint angles (7,)
        q_ctrl = np.array([js.position[n2i[name]] for name in JOINT_NAMES])

        # Build full pinocchio q (neutral for joints not in JOINT_NAMES)
        q = pin.neutral(self.model)
        for i, qi in enumerate(self.q_indices):
            q[qi] = q_ctrl[i]

        # FK + Jacobian
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        T = self.data.oMf[self.frame_id]
        ee_pos = T.translation.copy()
        pq = pin.Quaternion(T.rotation)
        ee_quat = np.array([pq.w, pq.x, pq.y, pq.z])  # wxyz

        J_full = pin.computeFrameJacobian(
            self.model, self.data, q, self.frame_id, pin.LOCAL_WORLD_ALIGNED
        )
        J = J_full[:, self.v_indices]   # (6, 7)
        if POSITION_ONLY:
            J = J[:3, :]

        # Target pose
        p = tgt.pose
        tgt_pos  = np.array([p.position.x, p.position.y, p.position.z])
        tgt_quat = np.array([p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z])

        # Pose error → DLS → new joint command
        pos_err, aa_err = compute_pose_error(ee_pos, ee_quat, tgt_pos, tgt_quat)
        pose_error = np.concatenate([pos_err, aa_err]) if not POSITION_ONLY else pos_err
        q_cmd = q_ctrl + compute_dls(pose_error, J)

        # Publish
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