#!/usr/bin/env python3
from threading import Lock
from typing import Optional

import numpy as np
from numpy.linalg import norm

import pinocchio as pin
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────
URDF_PATH   = "/home/neubility-sim/openarm_ros2_ws/src/openarm_description/urdf/robot/openarm_bimanual.urdf"
EE_FRAME    = "openarm_left_hand_tcp"
JOINT_NAMES = [f"openarm_left_joint{i}" for i in range(1, 8)]

# CLIK parameters (ref: gepettoweb.laas.fr pinocchio IK example)
IK_ITER     = 1000     # iterations per control loop call
DT          = 1.    # step size
DAMP        = 1e-6    # damping (λ in JJᵀ + λI)
EPS         = 1e-4   # early-exit convergence threshold
MAX_JOINT_V = 2.0    # max |Δq| per iteration (rad)
POSITION_ONLY = False

# ee_tcp (USD/training) has local Rx(π) from link7; hand_tcp (URDF) has rpy="0 0 0"
# → R_hand_desired = R_ee_desired @ Rx(π)
_RX_PI = np.diag([-1.0, 1.0, -1.0])

CONTROL_RATE      = 120.0
JOINT_STATE_TOPIC = "/joint_states"
TARGET_POSE_TOPIC = "/left/ik_target_pose"
PUBLISH_TOPIC     = "/arm/command"


# ─────────────────────────────────────────
# ROS2 node
# ─────────────────────────────────────────
class DifferentialIKNode(Node):
    def __init__(self):
        super().__init__("differential_ik_node")

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

        self._lock = Lock()
        self.latest_joint_state: Optional[JointState] = None
        self.latest_target_pose: Optional[PoseStamped] = None
        self.name_to_idx: dict = {}

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
            # Current joint positions
            q_ctrl = np.array([js.position[n2i[name]] for name in JOINT_NAMES],
                               dtype=np.float64)

            # SE3 target from PoseStamped (xyzw quaternion)
            p = tgt.pose
            tgt_pos = np.array([p.position.x, p.position.y, p.position.z])
            tgt_R   = pin.Quaternion(np.array([
                p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w
            ])).matrix()
            oMdes = pin.SE3(tgt_R, tgt_pos)
            
            # ── CLIK (ref: gepettoweb.laas.fr pinocchio IK example) ─────────
            q_cmd = q_ctrl.copy()
            for _ in range(IK_ITER):
                q = pin.neutral(self.model)
                for i, qi in enumerate(self.joint_q_idx):
                    q[qi] = q_cmd[i]

                pin.forwardKinematics(self.model, self.data, q)
                pin.updateFramePlacements(self.model, self.data)

                # SE3 log error in LOCAL frame (same as reference)
                dMf = oMdes.actInv(self.data.oMf[self.ee_frame_id])
                err = pin.log(dMf).vector   # [pos_err(3), rot_err(3)]
           
                if norm(err) < EPS:
                    break

                # Frame Jacobian in LOCAL frame — consistent with log error
                J_full = pin.computeFrameJacobian(
                    self.model, self.data, q, self.ee_frame_id, pin.LOCAL
                )
                J = J_full[:, self.joint_v_idx]  # select only controlled joints

                if POSITION_ONLY:
                    err = err[:3]
                    J   = J[:3]

                n = J.shape[0]
                # v = -Jᵀ (JJᵀ + λI)⁻¹ err
                v = -J.T @ np.linalg.solve(J @ J.T + DAMP * np.eye(n), err)
                v = np.clip(v, -MAX_JOINT_V, MAX_JOINT_V)
              
                q_cmd = q_cmd + v * DT
            # ────────────────────────────────────────────────────────────────

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
