#!/usr/bin/env python3
"""
Standalone pinocchio IK worker process.
No ROS2 / cv_bridge / rclpy — isolated from boost version conflicts.
Spawned by DifferentialIKControllerProxy.
"""
import sys

_ros_sp = "/opt/ros/humble/lib/python3.10/site-packages"
if _ros_sp in sys.path:
    sys.path.remove(_ros_sp)

import os
import numpy as np
import pinocchio as pin


def _quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


def _normalize_quat(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    return q / n if n > 1e-12 else q


def _quat_error_axis_angle(q_cur: np.ndarray, q_tgt: np.ndarray) -> np.ndarray:
    def q_mul(a, b):
        aw, ax, ay, az = a
        bw, bx, by, bz = b
        return np.array([
            aw*bw - ax*bx - ay*by - az*bz,
            aw*bx + ax*bw + ay*bz - az*by,
            aw*by - ax*bz + ay*bw + az*bx,
            aw*bz + ax*by - ay*bx + az*bw,
        ])
    q_cur_inv = np.array([q_cur[0], -q_cur[1], -q_cur[2], -q_cur[3]])
    q_err = q_mul(q_tgt, q_cur_inv)
    if q_err[0] < 0:
        q_err = -q_err
    angle = 2.0 * np.arctan2(np.linalg.norm(q_err[1:]), q_err[0])
    axis = q_err[1:] / (np.linalg.norm(q_err[1:]) + 1e-12)
    return angle * axis


def worker_main(conn, urdf_path, ee_frame, joint_names, ik_kwargs):
    """Entry point for the spawned subprocess."""
    urdf = os.path.expanduser(urdf_path)
    model = pin.buildModelFromUrdf(urdf)
    data = model.createData()

    if not model.existFrame(ee_frame):
        conn.send(('init_err', f"Frame '{ee_frame}' not found in URDF"))
        conn.close()
        return

    ee_frame_id = model.getFrameId(ee_frame)
    q_indices, v_indices = [], []
    for name in joint_names:
        jid = model.getJointId(name)
        q_indices.append(model.joints[jid].idx_q)
        v_indices.append(model.joints[jid].idx_v)

    ik_method   = ik_kwargs.get('ik_method', 'dls')
    lambda_val  = float(ik_kwargs.get('lambda_val', 0.01))
    max_delta_q = float(ik_kwargs.get('max_delta_q', 10.0))
    position_only = bool(ik_kwargs.get('position_only', False))

    conn.send(('ready', None))

    while True:
        try:
            msg = conn.recv()
        except EOFError:
            break
        if msg is None:
            break

        joint_positions, target_pos, target_quat_wxyz = msg
        try:
            q_full = pin.neutral(model)
            for qi, qv in zip(q_indices, joint_positions):
                q_full[qi] = float(qv)

            pin.forwardKinematics(model, data, q_full)
            pin.updateFramePlacements(model, data)

            frame_pose = data.oMf[ee_frame_id]
            ee_pos = frame_pose.translation.copy()
            ee_quat = _quat_xyzw_to_wxyz(
                pin.Quaternion(frame_pose.rotation).coeffs().copy()
            )

            J6 = pin.computeFrameJacobian(
                model, data, q_full, ee_frame_id,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            ).copy()
            J_ctrl = J6[:, v_indices]

            pos_err = target_pos - ee_pos
            if position_only:
                delta_x = pos_err
                J = J_ctrl[:3, :]
            else:
                rot_err = _quat_error_axis_angle(
                    ee_quat, _normalize_quat(target_quat_wxyz)
                )
                delta_x = np.concatenate([pos_err, rot_err])
                J = J_ctrl

            if ik_method == 'dls':
                m = J.shape[0]
                reg = (lambda_val ** 2) * np.eye(m, dtype=np.float64)
                delta_q = J.T @ np.linalg.solve(J @ J.T + reg, delta_x)
            else:
                delta_q = np.linalg.pinv(J) @ delta_x

            current_q = np.asarray(joint_positions, dtype=np.float64)
            delta_q = np.clip(delta_q, -max_delta_q, max_delta_q)
            conn.send(('ok', current_q + delta_q))

        except Exception as e:
            conn.send(('err', str(e)))

    conn.close()
