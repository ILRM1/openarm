#!/usr/bin/env python3
import math
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
URDF_PATH       = "/home/neubility-sim/openarm_ros2_ws/src/openarm_description/urdf/robot/openarm_bimanual.urdf"
EE_FRAME        = "openarm_left_hand_tcp"
JOINT_NAMES     = [f"openarm_left_joint{i}" for i in range(1, 8)]
IK_METHOD       = "dls"
MIN_SINGULAR    = 0.1
K_VAL = 1.
LAMBDA_VAL = 0.01
MAX_DELTA_Q = 0.5   # rad per control step
POSITION_ONLY   = False
CONTROL_RATE    = 10.0      # Hz

JOINT_STATE_TOPIC  = "/joint_states"
TARGET_POSE_TOPIC  = "/left/ik_target_pose"
PUBLISH_TOPIC      = "/arm/command"


# ─────────────────────────────────────────
# Quaternion helpers
# ─────────────────────────────────────────
def quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)

def normalize(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    return q / n if n > 1e-12 else np.array([1., 0., 0., 0.])

def quat_error_axis_angle(cur_wxyz: np.ndarray, tgt_wxyz: np.ndarray) -> np.ndarray:
    qc = normalize(cur_wxyz)
    qt = normalize(tgt_wxyz)

    # q_err = q_target * conj(q_current)
    w1, x1, y1, z1 = qt
    w2, x2, y2, z2 = qc[0], -qc[1], -qc[2], -qc[3]

    q_err = normalize(np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=np.float64))

    # shortest path
    if q_err[0] < 0.0:
        q_err = -q_err

    w = np.clip(q_err[0], -1.0, 1.0)
    v = q_err[1:]
    v_norm = np.linalg.norm(v)

    if v_norm < 1e-9:
        return np.zeros(3, dtype=np.float64)

    angle = 2.0 * math.atan2(v_norm, w)

    # angle을 [-pi, pi] 범위로 정리
    if angle > math.pi:
        angle -= 2.0 * math.pi

    return v / v_norm * angle


# ─────────────────────────────────────────
# IK solver (numpy only)
# ─────────────────────────────────────────
def solve_ik_dls(jacobian: np.ndarray, delta_x: np.ndarray, lam: float = 0.01) -> np.ndarray:
    m = jacobian.shape[0]
    return jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + lam**2 * np.eye(m), delta_x)


# ─────────────────────────────────────────
# ROS2 node
# ─────────────────────────────────────────
class DifferentialIKNode(Node):
    def __init__(self):
        super().__init__("differential_ik_node")

        # Pinocchio model
        self.model = pin.buildModelFromUrdf(URDF_PATH)
        self.data  = self.model.createData()

        if not self.model.existFrame(EE_FRAME):
            raise ValueError(f"EE frame '{EE_FRAME}' not found in URDF")
        self.ee_frame_id = self.model.getFrameId(EE_FRAME)

        self.joint_q_idx = []
        self.joint_v_idx = []
        for name in JOINT_NAMES:
            jid = self.model.getJointId(name)
            self.joint_q_idx.append(self.model.joints[jid].idx_q)
            self.joint_v_idx.append(self.model.joints[jid].idx_v)

        # State
        self._lock = Lock()
        self.latest_joint_state: Optional[JointState] = None
        self.latest_target_pose: Optional[PoseStamped] = None
        self.name_to_idx = {}

        # ROS2
        self.create_subscription(JointState, JOINT_STATE_TOPIC, self._joint_cb, 10)
        self.create_subscription(PoseStamped, TARGET_POSE_TOPIC, self._target_cb, 10)
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

    def _control_loop(self):
        with self._lock:
            js  = self.latest_joint_state
            tgt = self.latest_target_pose
            n2i = self.name_to_idx

        if js is None or tgt is None:
            return

        try:
            # Build full q
            q = pin.neutral(self.model)
            q_ctrl = np.zeros(len(JOINT_NAMES))
            for i, (name, qi) in enumerate(zip(JOINT_NAMES, self.joint_q_idx)):
                val = js.position[n2i[name]]
                q[qi] = val
                q_ctrl[i] = val

            # Forward kinematics + Jacobian
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)

            frame = self.data.oMf[self.ee_frame_id]
            ee_pos     = frame.translation.copy()
            ee_quat    = quat_xyzw_to_wxyz(pin.Quaternion(frame.rotation).coeffs().copy())
            
            J6 = pin.computeFrameJacobian(
                self.model, self.data, q,
                self.ee_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            )
            J = J6[:, self.joint_v_idx]

            # Target
            p = tgt.pose
            tgt_pos  = np.array([p.position.x, p.position.y, p.position.z])
            tgt_quat = normalize(np.array([
                p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z
            ]))
            
            # IK
            pos_err = tgt_pos - ee_pos
            rot_err = quat_error_axis_angle(ee_quat, tgt_quat)
            delta_x = np.concatenate([pos_err, rot_err]) if not POSITION_ONLY else pos_err

            #pos_err = np.clip(pos_err, -0.02, 0.02)
            #rot_err = np.clip(rot_err, -0.06, 0.06)
            delta_x = K_VAL * np.concatenate([pos_err, 0.3 * rot_err])

            J_use   = J if not POSITION_ONLY else J[:3]

            delta_q = solve_ik_dls(J_use, delta_x, LAMBDA_VAL)
            delta_q = np.clip(delta_q, -MAX_DELTA_Q, MAX_DELTA_Q)
            q_cmd   = q_ctrl + delta_q

            # Publish
            out = JointState()
            out.header.stamp = self.get_clock().now().to_msg()
            out.name     = JOINT_NAMES
            out.position = q_cmd.tolist()
            self.pub.publish(out)

        except Exception as e:
            self.get_logger().error(f"IK error: {e}")


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
