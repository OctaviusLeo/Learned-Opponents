"""GPU navigation environment on mjlab / MuJoCo Warp (tasks 5.1, 6.1).

``num_envs`` worlds step in parallel in one ``mjlab.sim.Simulation``. Each
control step (``control_dt_s``) applies the (latency-delayed) wheel command,
runs ``physics_dt_s`` substeps, decides termination from the fresh state,
resets finished worlds, refreshes kinematics and rangefinders once with
``forward()``, and builds the (latency-delayed) Obs_Action_Spec observation
with the shared array-API code in ``spec/``, ``obstacles``, and ``rules``.

Per-episode content comes from banks sampled up front from the run seed:
obstacle layouts (``obstacles.sample_layouts``) and domain parameters
(``domain_rand.sample_params``). World ``i`` uses bank entry
``(episode_count[i] * num_envs + i) % bank_size``, so the draws are
reproducible and need no host round trip. Ray noise uses a seeded torch
generator. All tensors live on the GPU; torch runs on Warp's CUDA stream so
reads never race the physics kernels.

Demo (success, collision, timeout, each tagged with its reason):
    python -m learned_opponents.env.nav_env --demo
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Any, Mapping

import mujoco
import numpy as np
import torch
import warp as wp
from mjlab.sim.sim import MujocoCfg, Simulation, SimulationCfg

from learned_opponents.env import obstacles as obs_lib
from learned_opponents.env import rules
from learned_opponents.env import torch_xp as txp
from learned_opponents.env.model import box_body, build_mjcf, cyl_body, PARK_Z
from learned_opponents.randomization import domain_rand as dr
from learned_opponents.spec import obs_action_spec as spec

DR_FIELDS = ("body_mass", "body_inertia", "geom_friction", "actuator_gainprm",
             "actuator_biasprm", "actuator_forcerange", "geom_size")


@dataclass
class StepInfo:
    reason: torch.Tensor          # (N,) int: rules.NONE / SUCCESS / COLLISION / TIMEOUT
    time_outs: torch.Tensor       # (N,) bool: ended by the step limit
    terms: dict[str, torch.Tensor]
    episode_steps: torch.Tensor   # (N,) steps taken in the episode that just ended (0 if running)


class NavEnv:
    """Batched differential-drive navigation task."""

    def __init__(self, task: Mapping[str, Any], num_envs: int, seed: int, variant: str = "baseline",
                 dr_ranges: Mapping[str, dr.ParamRange] | None = None, device: str = "cuda:0",
                 bank_size: int = 2**18, max_latency: int = 6, auto_reset: bool = True,
                 sim_iterations: int = 20):
        self.task = dict(task)
        self.cfg = spec.as_spec_config(task)
        self.num_envs = int(num_envs)
        self.seed = int(seed)
        self.variant = variant
        self.device = device
        self.auto_reset = auto_reset
        self.n_rays = self.cfg.n_rays
        self.obs_dim = self.cfg.obs_dim
        self.substeps = int(round(task["control_dt_s"] / task["physics_dt_s"]))
        self.episode_limit = int(task["episode_limit"])
        self.k = int(task["n_obstacles"][1])
        self.max_latency = int(max_latency)
        self.weights = dict(task["reward"])

        wp.config.quiet = True
        wp.init()
        torch.cuda.set_device(torch.device(device))
        # One stream for Warp kernels, CUDA-graph replays, and torch ops.
        torch.cuda.set_stream(torch.cuda.ExternalStream(wp.get_stream(device).cuda_stream))

        self.mj_model = mujoco.MjModel.from_xml_string(build_mjcf(task))
        sim_cfg = SimulationCfg(nconmax=8, njmax=16, mujoco=MujocoCfg(
            timestep=float(task["physics_dt_s"]), integrator="implicitfast", cone="pyramidal",
            iterations=sim_iterations, ls_iterations=10))
        self.sim = Simulation(self.num_envs, sim_cfg, model=self.mj_model, device=device)
        self.sim.expand_model_fields(DR_FIELDS)
        self._index_model()

        n = self.num_envs
        z = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt, device=device)  # noqa: E731
        self.step_count = z(n, dt=torch.long)
        self.episode_count = z(n, dt=torch.long)
        self.prev_dist = z(n)
        self.goal = z(n, 2)
        self.kind = z(n, self.k, dt=torch.int8)
        self.center = z(n, self.k, 2)
        self.size = z(n, self.k)
        self.params = {p: z(n) for p in dr.PARAMS}
        self.act_buf = z(n, self.max_latency + 1, 2)
        self.obs_buf = z(n, self.max_latency + 1, self.obs_dim)
        self.obs = z(n, self.obs_dim)
        self._arange = torch.arange(n, device=device)
        self.noise_gen = torch.Generator(device=device).manual_seed(self.seed)
        self._override: dict[str, float] | None = None

        self.dr_ranges = dict(dr_ranges) if dr_ranges is not None else None
        if variant == "randomized" and self.dr_ranges is None:
            raise ValueError("the randomized variant needs dr_ranges")
        self._make_banks(bank_size)

    # ------------------------------------------------------------------ setup

    def _index_model(self) -> None:
        m = self.mj_model
        bid = lambda name: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)  # noqa: E731
        gid = lambda name: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)  # noqa: E731
        dev = self.device
        self.robot_bodies = torch.tensor([bid("robot"), bid("wheel_l"), bid("wheel_r")], device=dev)
        self.wheel_geoms = torch.tensor([gid("wheel_l"), gid("wheel_r")], device=dev)
        self.box_geoms = torch.tensor([gid(box_body(j)) for j in range(self.k)], device=dev)
        self.cyl_geoms = torch.tensor([gid(cyl_body(j)) for j in range(self.k)], device=dev)
        self.box_mocap = torch.tensor([int(m.body_mocapid[bid(box_body(j))]) for j in range(self.k)], device=dev)
        self.cyl_mocap = torch.tensor([int(m.body_mocapid[bid(cyl_body(j))]) for j in range(self.k)], device=dev)
        self.half_h = float(self.task["obstacle_height_m"]) / 2.0
        mf = self.sim.model
        self.nom_mass = mf.body_mass[0, self.robot_bodies].clone()
        self.nom_inertia = mf.body_inertia[0, self.robot_bodies].clone()
        self.nom_gain = mf.actuator_gainprm[0, :, 0].clone()
        self.nom_bias = mf.actuator_biasprm[0, :, 2].clone()
        self.nom_force = mf.actuator_forcerange[0].clone()
        self.nom_friction = float(self.task["wheel_friction"])
        self.nom_box_size = mf.geom_size[0, self.box_geoms].clone()
        self.nom_cyl_size = mf.geom_size[0, self.cyl_geoms].clone()

    def _make_banks(self, bank_size: int) -> None:
        lay = obs_lib.sample_layouts(dr.make_rng(self.seed, "layouts"), bank_size, self.task)
        self._set_bank(lay)
        if self.variant == "randomized":
            p = dr.sample_params(dr.make_rng(self.seed, "domain_rand"), bank_size, self.dr_ranges,
                                 "randomized")
        else:
            base = self.dr_ranges or dr.parse_ranges({k: {"nominal": v, "range": [v, v]} for k, v in
                                                      zip(dr.PARAMS, (1.0, 1.0, 1.0, 0.0, 0, 0))})
            p = dr.nominal_params(bank_size, base)
        self.param_bank = {k: torch.as_tensor(v, dtype=torch.float32, device=self.device) for k, v in p.items()}

    def _set_bank(self, lay: obs_lib.Layouts) -> None:
        t = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=self.device)  # noqa: E731
        self.bank = {"kind": t(lay.kind, torch.int8), "center": t(lay.center), "size": t(lay.size),
                     "start": t(lay.start), "goal": t(lay.goal)}
        self.bank_size = len(lay)

    # ------------------------------------------------------------- overrides

    def set_param_override(self, values: Mapping[str, float] | None) -> None:
        """Fix domain parameters for every episode (evaluation and robustness cells)."""
        self._override = None if values is None else {k: float(values[k]) for k in dr.PARAMS}

    def use_layouts(self, layouts: obs_lib.Layouts) -> None:
        """Replace the layout bank (evaluation and transfer replay); world i gets entry i."""
        self._set_bank(layouts)

    # ------------------------------------------------------------------ state

    def robot_state(self):
        """(xy, yaw, forward speed, yaw rate) from the fresh qpos/qvel."""
        q = self.sim.data.qpos
        v = self.sim.data.qvel
        w, x, y, zq = q[:, 3], q[:, 4], q[:, 5], q[:, 6]
        yaw = torch.atan2(2.0 * (w * zq + x * y), 1.0 - 2.0 * (y * y + zq * zq))
        fwd = torch.cos(yaw) * v[:, 0] + torch.sin(yaw) * v[:, 1]
        return q[:, 0:2], yaw, fwd, v[:, 5]

    def raw_rays(self) -> torch.Tensor:
        return self.sim.data.sensordata[:, : self.n_rays]

    def goal_distance(self, xy: torch.Tensor) -> torch.Tensor:
        return torch.linalg.vector_norm(self.goal - xy, dim=-1)

    def collided(self, xy: torch.Tensor) -> torch.Tensor:
        return obs_lib.collides_xp(txp, xy, float(self.task["robot_radius_m"]), self.kind, self.center,
                                   self.size, float(self.task["arena_size_m"]))

    # ------------------------------------------------------------------ reset

    def _episode_values(self, mask: torch.Tensor):
        idx = (self.episode_count * self.num_envs + self._arange) % self.bank_size
        lay = {k: v[idx] for k, v in self.bank.items()}
        if self._override is not None:
            par = {k: torch.full((self.num_envs,), v, device=self.device) for k, v in self._override.items()}
        else:
            par = {k: v[idx] for k, v in self.param_bank.items()}
        return lay, par

    def _reset_worlds(self, mask: torch.Tensor) -> None:
        """Start a new episode in every world where ``mask`` is True."""
        lay, par = self._episode_values(mask)
        m2 = mask[:, None]
        m3 = mask[:, None, None]
        self.sim.reset(mask)

        # Obstacles: move the active body of each slot in, park the other, resize both.
        is_box = (lay["kind"] == obs_lib.BOX)
        is_cyl = (lay["kind"] == obs_lib.CYLINDER)
        park = torch.tensor([0.0, 0.0, PARK_Z], device=self.device)
        pos = torch.cat([lay["center"], torch.full_like(lay["size"][..., None], self.half_h)], dim=-1)
        mp = self.sim.data.mocap_pos
        mp[:, self.box_mocap] = torch.where(m3, torch.where(is_box[..., None], pos, park), mp[:, self.box_mocap])
        mp[:, self.cyl_mocap] = torch.where(m3, torch.where(is_cyl[..., None], pos, park), mp[:, self.cyl_mocap])
        half = (lay["size"] * 0.5).clamp_min(1e-3)
        gs = self.sim.model.geom_size
        box_size = torch.stack([half, half, torch.full_like(half, self.half_h)], dim=-1)
        cyl_size = torch.stack([half, torch.full_like(half, self.half_h), torch.zeros_like(half)], dim=-1)
        gs[:, self.box_geoms] = torch.where(m3, box_size, gs[:, self.box_geoms])
        gs[:, self.cyl_geoms] = torch.where(m3, cyl_size, gs[:, self.cyl_geoms])

        # Domain parameters (scale factors on the nominal model values).
        mf = self.sim.model
        ms = par["mass_scale"][:, None]
        mf.body_mass[:, self.robot_bodies] = torch.where(m2, self.nom_mass * ms, mf.body_mass[:, self.robot_bodies])
        mf.body_inertia[:, self.robot_bodies] = torch.where(
            m3, self.nom_inertia * ms[..., None], mf.body_inertia[:, self.robot_bodies])
        fr = (self.nom_friction * par["friction_scale"])[:, None]
        mf.geom_friction[:, self.wheel_geoms, 0] = torch.where(m2, fr, mf.geom_friction[:, self.wheel_geoms, 0])
        st = par["motor_strength_scale"][:, None]
        mf.actuator_gainprm[:, :, 0] = torch.where(m2, self.nom_gain * st, mf.actuator_gainprm[:, :, 0])
        mf.actuator_biasprm[:, :, 2] = torch.where(m2, self.nom_bias * st, mf.actuator_biasprm[:, :, 2])
        mf.actuator_forcerange[:] = torch.where(m3, self.nom_force * st[..., None], mf.actuator_forcerange[:])
        for k in dr.PARAMS:
            self.params[k] = torch.where(mask, par[k], self.params[k])

        # Robot pose: wheels on the floor, yaw from the layout, at rest.
        q = self.sim.data.qpos
        start = lay["start"]
        new_q = torch.zeros_like(q)
        new_q[:, 0:2] = start[:, 0:2]
        new_q[:, 2] = float(self.task["wheel_radius_m"])
        new_q[:, 3] = torch.cos(start[:, 2] / 2)
        new_q[:, 6] = torch.sin(start[:, 2] / 2)
        q[:] = torch.where(m2, new_q, q[:])
        self.sim.data.qvel[:] = torch.where(m2, 0.0, self.sim.data.qvel[:])
        self.sim.data.ctrl[:] = torch.where(m2, 0.0, self.sim.data.ctrl[:])

        self.goal = torch.where(m2, lay["goal"], self.goal)
        self.kind = torch.where(m2, lay["kind"], self.kind)
        self.center = torch.where(m3, lay["center"], self.center)
        self.size = torch.where(m2, lay["size"], self.size)
        self.prev_dist = torch.where(mask, self.goal_distance(new_q[:, 0:2]), self.prev_dist)
        self.step_count = torch.where(mask, 0, self.step_count)
        self.episode_count = self.episode_count + mask.long()
        self.act_buf = torch.where(m3, 0.0, self.act_buf)
        # Mass may have changed: refresh subtree masses (set_const_fixed, 0.1 ms for 4096
        # worlds). The solver's reference inverse weights (set_const_0, 11.7 ms) stay at the
        # nominal model's values; they set constraint regularization, not the dynamics,
        # which use body_mass and body_inertia directly.
        self.sim.recompute_constants(_SET_CONST_FIXED)

    def reset(self) -> torch.Tensor:
        """Reset every world (episode counters restart) and return the initial observation."""
        self.episode_count.zero_()
        mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._reset_worlds(mask)
        self.episode_count.zero_()   # the first episode of world i is bank entry i
        self.sim.forward()
        fresh = self._observe()
        self.obs_buf[:] = fresh[:, None, :]
        self.obs = fresh
        return self.obs

    # ------------------------------------------------------------------- step

    def _observe(self) -> torch.Tensor:
        xy, yaw, fwd, rate = self.robot_state()
        rays = self.raw_rays()
        sigma = self.params["sensor_noise_sigma_m"][:, None]
        noise = torch.randn(rays.shape, generator=self.noise_gen, device=self.device) * sigma
        rays = torch.where(rays >= 0, rays + noise, rays)
        return spec.build_observation_xp(txp, rays, xy, yaw, self.goal, fwd, rate, self.cfg)

    def step(self, actions: torch.Tensor):
        """Advance one control step. Returns ``(obs, reward, done, StepInfo)``."""
        a = torch.clip(actions.to(self.device, torch.float32), -1.0, 1.0)
        self.act_buf = torch.roll(self.act_buf, 1, dims=1)
        self.act_buf[:, 0] = a
        lat_a = self.params["action_latency_steps"].long().clamp(0, self.max_latency)
        applied = self.act_buf[self._arange, lat_a]
        self.sim.data.ctrl[:] = applied * self.cfg.max_wheel_speed
        for _ in range(self.substeps):
            self.sim.step()

        xy, _, _, _ = self.robot_state()
        self.step_count = self.step_count + 1
        dist = self.goal_distance(xy)
        reason = rules.termination_reason_xp(txp, dist, self.collided(xy), self.step_count,
                                             float(self.task["arrival_tolerance_m"]), self.episode_limit)
        terms = rules.reward_terms_xp(txp, self.prev_dist, dist, reason)
        reward = rules.combine_reward(terms, self.weights)
        self.prev_dist = dist
        done = reason != rules.NONE
        info = StepInfo(reason=reason, time_outs=reason == rules.TIMEOUT, terms=terms,
                        episode_steps=torch.where(done, self.step_count, 0))
        if self.auto_reset and bool(done.any()):
            self._reset_worlds(done)
        self.sim.forward()
        fresh = self._observe()
        if self.auto_reset:
            self.obs_buf = torch.where(done[:, None, None], fresh[:, None, :], self.obs_buf)
        self.obs_buf = torch.roll(self.obs_buf, 1, dims=1)
        self.obs_buf[:, 0] = fresh
        lat_o = self.params["observation_latency_steps"].long().clamp(0, self.max_latency)
        self.obs = self.obs_buf[self._arange, lat_o]
        return self.obs, reward, done, info


def _set_const_level():
    from mjlab.managers.event_manager import RecomputeLevel
    return RecomputeLevel.set_const_fixed


_SET_CONST_FIXED = _set_const_level()


# ----------------------------------------------------------------------- demo

def _demo(device: str = "cuda:0") -> int:
    """Three scripted episodes, one per termination reason."""
    from learned_opponents.config import load_task

    task = load_task()
    env = NavEnv(task, num_envs=3, seed=0, device=device, bank_size=3, auto_reset=False)
    k = env.k
    kind = np.zeros((3, k), np.int8)
    center = np.zeros((3, k, 2))
    size = np.zeros((3, k))
    kind[1, 0], center[1, 0], size[1, 0] = obs_lib.BOX, (1.0, 0.0), 0.6   # box straight ahead of env 1
    start = np.array([[-2.0, -2.0, 0.3], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    goal = np.array([[2.5, 2.0], [3.0, 0.0], [3.0, 3.0]])
    env.use_layouts(obs_lib.Layouts(kind, center, size, start, goal))
    env.reset()
    names = ["go-to-goal controller", "drive straight into a box", "spin in place"]
    first = [None, None, None]
    for t in range(env.episode_limit):
        xy, yaw, _, _ = env.robot_state()
        d = env.goal - xy
        err = torch.atan2(d[:, 1], d[:, 0]) - yaw
        err = torch.atan2(torch.sin(err), torch.cos(err))
        steer = torch.clip(2.0 * err, -1, 1)
        go = torch.clip(1.0 - 0.8 * steer.abs(), 0.2, 1.0)
        act = torch.stack([go - steer * 0.5, go + steer * 0.5], dim=-1)
        act[1] = torch.tensor([0.6, 0.6])
        act[2] = torch.tensor([-0.5, 0.5])
        _, _, done, info = env.step(act)
        for i in range(3):
            if first[i] is None and bool(done[i]):
                first[i] = (rules.REASON_NAMES[int(info.reason[i])], t + 1)
        if all(f is not None for f in first):
            break
    ok = [f is not None and f[0] == want for f, want in zip(first, ["success", "collision", "timeout"])]
    for name, f, good in zip(names, first, ok):
        print(f"{name:28s} -> {f[0] if f else 'running'} after {f[1] if f else '-'} steps "
              f"({'as expected' if good else 'UNEXPECTED'})")
    return 0 if all(ok) else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    raise SystemExit(_demo(args.device) if args.demo else 0)
