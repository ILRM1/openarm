"""
dual_arm_ik_node.py — OpenArm v10 양팔 6D IK (Pinocchio + ROS2)
─────────────────────────────────────────────────────────────────
URDF:  urdf/robot/v10.urdf.xacro  (bimanual:=true)
Joint: left_joint[1-7],  right_joint[1-7]
EE:    left_hand_link  /  right_hand_link  (프레임 확인 후 수정)

Subscribe:
  /left/ik_target_pose    geometry_msgs/PoseStamped
  /right/ik_target_pose   geometry_msgs/PoseStamped
  /joint_states           sensor_msgs/JointState

Publish:
  /left/ik_solution              sensor_msgs/JointState
  /right/ik_solution             sensor_msgs/JointState
  /dual_arm/joint_trajectory     trajectory_msgs/JointTrajectory
  /left/ik_status                std_msgs/String
  /right/ik_status               std_msgs/String
"""

import numpy as np
import threading
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

import pinocchio as pin

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import String
from builtin_interfaces.msg import Duration


# ─────────────────────────────────────────────────────────────────
# OpenArm v10 bimanual 설정값
# → setup.bash 스크립트로 프레임 이름 확인 후 필요 시 수정
# ─────────────────────────────────────────────────────────────────

OPENARM_URDF = "~/isaac_ws/openarm_bimanual.urdf"   # URDF 경로

LEFT_EE_FRAME  = "openarm_left_hand_tcp"    # 확인 후 수정
RIGHT_EE_FRAME = "openarm_right_hand_tcp"   # 확인 후 수정

LEFT_JOINTS  = [f"openarm_left_joint{i}"  for i in range(1, 8)]   # joint1~7
RIGHT_JOINTS = [f"openarm_right_joint{i}" for i in range(1, 8)]


# ─────────────────────────────────────────────────────────────────
# 1. 자료 구조
# ─────────────────────────────────────────────────────────────────

class Side(Enum):
    LEFT  = "left"
    RIGHT = "right"


@dataclass
class ArmConfig:
    side:        Side
    ee_frame:    str
    joint_names: list[str]
    q_indices:   list[int] = field(default_factory=list)


@dataclass
class ArmState:
    q:      np.ndarray     = field(default_factory=lambda: np.zeros(7))
    target: Optional[pin.SE3] = None
    lock:   threading.Lock = field(default_factory=threading.Lock)


# ─────────────────────────────────────────────────────────────────
# 2. 6D IK Solver (Damped Least Squares)
# ─────────────────────────────────────────────────────────────────

class IKSolver6D:
    def __init__(
        self,
        model:    pin.Model,
        ee_frame: str,
        max_iter: int   = 300,
        eps:      float = 1e-4,
        damping:  float = 1e-6,
        dt:       float = 0.1,
    ):
        self.model    = model
        self.data     = model.createData()   # 팔별 독립 인스턴스
        self.ee_id    = model.getFrameId(ee_frame)
        self.max_iter = max_iter
        self.eps      = eps
        self.damping  = damping
        self.dt       = dt

    def _clamp(self, q):
        return np.clip(q, self.model.lowerPositionLimit, self.model.upperPositionLimit)

    def fk(self, q: np.ndarray) -> pin.SE3:
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        return self.data.oMf[self.ee_id].copy()

    def solve(self, target: pin.SE3, q_init: np.ndarray):
        q = q_init.copy()
        for _ in range(self.max_iter):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            err      = pin.log6(self.data.oMf[self.ee_id].actInv(target)).vector
            err_norm = float(np.linalg.norm(err))

            if err_norm < self.eps:
                return self._clamp(q), True, err_norm

            J   = pin.computeFrameJacobian(
                self.model, self.data, q, self.ee_id,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )
            JJT = J @ J.T + (self.damping ** 2) * np.eye(6)
            dq  = J.T @ np.linalg.solve(JJT, err)
            q   = self._clamp(pin.integrate(self.model, q, dq * self.dt))

        return self._clamp(q), False, float(np.linalg.norm(err))


# ─────────────────────────────────────────────────────────────────
# 3. Dual-Arm IK Node
# ─────────────────────────────────────────────────────────────────

class OpenArmDualIKNode(Node):

    def __init__(self):
        super().__init__("openarm_dual_ik_node")
        self._declare_params()
        self._load_robot()
        self._init_arms()
        self._setup_pubsub()
        self._start_ik_loop()
        self.get_logger().info("OpenArm DualArm IK Node ready ✓")

    # ── 파라미터 ──────────────────────────────────────────────

    def _declare_params(self):
        self.declare_parameter("urdf_path",       OPENARM_URDF)
        self.declare_parameter("left_ee_frame",   LEFT_EE_FRAME)
        self.declare_parameter("right_ee_frame",  RIGHT_EE_FRAME)
        self.declare_parameter("left_joint_names",  LEFT_JOINTS)
        self.declare_parameter("right_joint_names", RIGHT_JOINTS)
        self.declare_parameter("ik_rate_hz",      50.0)
        self.declare_parameter("trajectory_dt",   0.1)

    # ── 로봇 로드 ──────────────────────────────────────────────

    def _load_robot(self):
        import os
        urdf = os.path.expanduser(self.get_parameter("urdf_path").value)
        self.model = pin.buildModelFromUrdf(urdf)
        self.get_logger().info(
            f"OpenArm loaded: {self.model.nq} DOF | URDF: {urdf}"
        )

        # 프레임 목록 출력 (디버그용 — EE 이름 모를 때 참고)
        self.get_logger().debug("Available frames:")
        for i in range(self.model.nframes):
            self.get_logger().debug(f"  [{i}] {self.model.frames[i].name}")

    # ── 팔별 초기화 ────────────────────────────────────────────

    def _init_arms(self):
        q0 = pin.neutral(self.model)

        raw = {
            Side.LEFT: ArmConfig(
                side        = Side.LEFT,
                ee_frame    = self.get_parameter("left_ee_frame").value,
                joint_names = self.get_parameter("left_joint_names").value,
            ),
            Side.RIGHT: ArmConfig(
                side        = Side.RIGHT,
                ee_frame    = self.get_parameter("right_ee_frame").value,
                joint_names = self.get_parameter("right_joint_names").value,
            ),
        }

        self.solvers: dict[Side, IKSolver6D] = {}
        self.configs: dict[Side, ArmConfig]  = {}
        self.states:  dict[Side, ArmState]   = {}

        for side, cfg in raw.items():
            # EE 프레임 검증
            if not self.model.existFrame(cfg.ee_frame):
                candidates = [
                    self.model.frames[i].name
                    for i in range(self.model.nframes)
                    if any(k in self.model.frames[i].name.lower()
                           for k in ["hand","ee","tool","link7","flange"])
                ]
                self.get_logger().fatal(
                    f"EE frame '{cfg.ee_frame}' not found!\n"
                    f"  EE 후보: {candidates}\n"
                    f"  → launch 파라미터 {side.value}_ee_frame 을 수정하세요."
                )
                raise RuntimeError(f"EE frame not found: {cfg.ee_frame}")

            # q 인덱스 매핑
            for jname in cfg.joint_names:
                if self.model.existJointName(jname):
                    jid = self.model.getJointId(jname)
                    cfg.q_indices.append(self.model.joints[jid].idx_q)
                else:
                    self.get_logger().warn(f"[{side.value}] '{jname}' not in model")

            self.solvers[side] = IKSolver6D(self.model, cfg.ee_frame)
            self.configs[side] = cfg

            q_arm = np.array([q0[i] for i in cfg.q_indices])
            self.states[side] = ArmState(q=q_arm)

            self.get_logger().info(
                f"[{side.value:5s}] EE='{cfg.ee_frame}'  "
                f"joints={cfg.joint_names}"
            )

    # ── Pub / Sub ──────────────────────────────────────────────

    def _setup_pubsub(self):
        qos_be = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE, depth=1,
        )
        qos_r = QoSProfile(depth=10)

        self.create_subscription(
            JointState, "/joint_states", self._cb_joint_states, qos_be,
        )

        self._pubs: dict[str, any] = {}
        for side in Side:
            ns = side.value
            self.create_subscription(
                PoseStamped, f"/{ns}/ik_target_pose",
                lambda msg, s=side: self._cb_target(msg, s), qos_r,
            )
            self._pubs[f"{ns}/solution"] = self.create_publisher(
                JointState, f"/{ns}/ik_solution", qos_r)
            self._pubs[f"{ns}/status"] = self.create_publisher(
                String, f"/{ns}/ik_status", qos_r)

        self._pubs["dual/traj"] = self.create_publisher(
            JointTrajectory, "/dual_arm/joint_trajectory", qos_r)

    # ── 콜백 ──────────────────────────────────────────────────

    def _cb_target(self, msg: PoseStamped, side: Side):
        p, o = msg.pose.position, msg.pose.orientation
        R    = pin.Quaternion(o.w, o.x, o.y, o.z).toRotationMatrix()
        with self.states[side].lock:
            self.states[side].target = pin.SE3(R, np.array([p.x, p.y, p.z]))

    def _cb_joint_states(self, msg: JointState):
        name_to_pos = dict(zip(msg.name, msg.position))
        for side, cfg in self.configs.items():
            with self.states[side].lock:
                for i, jname in enumerate(cfg.joint_names):
                    if jname in name_to_pos:
                        self.states[side].q[i] = name_to_pos[jname]

    # ── IK 루프 ────────────────────────────────────────────────

    def _start_ik_loop(self):
        rate = self.get_parameter("ik_rate_hz").value
        self.create_timer(1.0 / rate, self._ik_tick)
        self.get_logger().info(f"IK loop: {rate} Hz")

    def _ik_tick(self):
        results = {}
        for side in Side:
            state, cfg, solver = (
                self.states[side], self.configs[side], self.solvers[side]
            )
            with state.lock:
                if state.target is None:
                    continue
                target = state.target
                q_arm  = state.q.copy()

            # 팔 관절 → 전체 q 벡터 재구성
            q_full = pin.neutral(self.model)
            for i, qidx in enumerate(cfg.q_indices):
                q_full[qidx] = q_arm[i]

            q_sol_full, ok, err = solver.solve(target, q_full)

            # 팔 관절만 추출해서 저장
            q_sol_arm = np.array([q_sol_full[i] for i in cfg.q_indices])
            with state.lock:
                state.q = q_sol_arm

            results[side] = (q_sol_arm, ok, err)
            self._pub_arm(side, q_sol_arm, ok, err)

        if len(results) == 2:
            self._pub_dual_traj(results)

    # ── 퍼블리시 ───────────────────────────────────────────────

    def _pub_arm(self, side: Side, q: np.ndarray, ok: bool, err: float):
        ns, cfg, now = side.value, self.configs[side], self.get_clock().now().to_msg()

        js = JointState()
        js.header.stamp = now
        js.name         = cfg.joint_names
        js.position     = q.tolist()
        self._pubs[f"{ns}/solution"].publish(js)

        st = String()
        st.data = f"{'OK' if ok else 'FAIL'} | err={err*1000:.3f}mm"
        self._pubs[f"{ns}/status"].publish(st)

    def _pub_dual_traj(self, results: dict):
        dt = self.get_parameter("trajectory_dt").value

        traj = JointTrajectory()
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.joint_names  = (
            self.configs[Side.LEFT].joint_names +
            self.configs[Side.RIGHT].joint_names
        )
        pt = JointTrajectoryPoint()
        pt.positions = (
            results[Side.LEFT][0].tolist() +
            results[Side.RIGHT][0].tolist()
        )
        pt.time_from_start = Duration(
            sec=int(dt), nanosec=int((dt % 1) * 1e9)
        )
        traj.points = [pt]
        self._pubs["dual/traj"].publish(traj)


# ─────────────────────────────────────────────────────────────────
# 4. Entry Point
# ─────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = OpenArmDualIKNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()