# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pointwise fixed-path retiming with analytical derivatives and optional IPOPT."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import minimize


class PointwiseTiming:
    """Fixed cubic geometry, pointwise dynamic inequalities and 32 speed controls."""

    def __init__(self, directory: Path, stride: int = 1, grid_count: int = 0, refine: bool = False) -> None:
        data = np.load(directory / 'fixed_path_attempt.npz')
        grid = data['path_grid'][::stride]
        self.path = data['path'][:, ::stride]
        self.coeff = data['torque_coefficients'][:, ::stride]
        if grid_count:
            import pinocchio as pin
            from .fixed_path_control import path_at
            buffers = np.load(directory / 'native_buffers.npz')
            lane = json.loads((directory / 'reconstruction.json').read_text())['selected_lane_by_position_match']
            count = buffers['controls'].shape[1]
            knots = np.arange(1, count + 3) / (count + 3)
            grid = np.unique(np.r_[np.linspace(0, 1, grid_count), knots - 1e-8, knots, knots + 1e-8])
            if refine:
                dense = np.load(directory / 'refinement_input_2049.npz')
                extra = []
                for name in ('qd', 'qdd', 'jerk'):
                    magnitude = np.abs(dense[name])
                    for joint in range(7):
                        index = int(np.argmax(magnitude[:, joint]))
                        extra.extend(dense['progress'][max(0,index-2):index+3])
                grid = np.unique(np.r_[grid, extra])
            self.path = np.array(path_at(grid, buffers, lane))
            root = Path(__file__).resolve().parents[2]
            model = pin.buildModelFromUrdf(str(root / 'curobo/content/assets/robot/franka_description/franka_panda.urdf'))
            inertia = pin.Inertia(.01, np.zeros(3), np.eye(3)*.0001)
            frame = model.frames[model.getFrameId('right_gripper', pin.BODY)]
            model.inertias[frame.parentJoint] -= frame.placement.act(inertia)
            for name in ('panda_link8', 'panda_leftfinger', 'panda_rightfinger', 'panda_hand'):
                frame = model.frames[model.getFrameId(name, pin.BODY)]
                model.inertias[frame.parentJoint] += frame.placement.act(inertia)
            state, data_pin = pin.neutral(model), model.createData()
            qi = [model.joints[model.getJointId(f'panda_joint{i}')].idx_q for i in range(1, 8)]
            vi = [model.joints[model.getJointId(f'panda_joint{i}')].idx_v for i in range(1, 8)]
            for name in ('panda_finger_joint1', 'panda_finger_joint2'):
                state[model.joints[model.getJointId(name)].idx_q] = .04
            self.coeff = np.empty((3, len(grid), 7))
            for index in range(len(grid)):
                state[qi] = self.path[0, index]
                vel, acc = np.zeros(model.nv), np.zeros(model.nv)
                gravity = pin.rnea(model, data_pin, state, vel, acc)[vi].copy()
                acc[vi] = self.path[1, index]
                accel = pin.rnea(model, data_pin, state, vel, acc)[vi].copy() - gravity
                vel[vi], acc[vi] = self.path[1, index], self.path[2, index]
                moving = pin.rnea(model, data_pin, state, vel, acc)[vi].copy() - gravity
                self.coeff[:, index] = accel, moving, gravity
        self.grid = grid
        self.original = data['initial_timing']
        self.initial = data['solution_timing'] * .997
        self.scale = float(np.median(self.original))
        basis = BSpline(np.r_[np.zeros(3), np.linspace(0, 1, 30), np.ones(3)], np.eye(32), 3)
        self.b = [basis(grid, nu=i) * self.scale for i in range(3)]
        nodes, weights = np.polynomial.legendre.leggauss(6)
        quadrature = (np.arange(511)[:, None] + (nodes[None, :] + 1) / 2) / 511
        self.q = basis(quadrature.ravel()) * self.scale
        self.qw = np.tile(weights / (2 * 511), 511)
        raw = np.load(directory / 'baseline_validation_states.npz')
        bounds = np.stack([raw[name + '_bounds'] for name in
                           ('velocity', 'acceleration', 'jerk', 'torque')])
        self.center = (bounds[:, 1] + bounds[:, 0]) / 2
        self.width = (bounds[:, 1] - bounds[:, 0]) * .475

    def objective(self, z: np.ndarray) -> float:
        """Integrate reciprocal speed with fixed Gauss quadrature."""
        return float(self.qw @ (self.q @ z) ** -.5)

    def gradient(self, z: np.ndarray) -> np.ndarray:
        """Differentiate quadrature with respect to scaled controls."""
        return self.q.T @ (-.5 * self.qw * (self.q @ z) ** -1.5)

    def values_jac(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return normalized pointwise margins and their analytical Jacobian."""
        x, dx, ddx = [b @ z for b in self.b]
        speed = np.sqrt(x)[:, None]
        q1, q2, q3 = self.path[1:]
        a, m, gravity = self.coeff
        h = q3 * x[:, None] + 1.5 * q2 * dx[:, None] + .5 * q1 * ddx[:, None]
        values = np.stack((q1 * speed, q2 * x[:, None] + .5 * q1 * dx[:, None],
                           speed * h, gravity + m * x[:, None] + .5 * a * dx[:, None]))
        b0, b1, b2 = (b[:, None, :] for b in self.b)
        deriv = np.stack((q1[:, :, None] * b0 / (2 * speed[:, :, None]),
                          q2[:, :, None] * b0 + .5 * q1[:, :, None] * b1,
                          h[:, :, None] * b0 / (2 * speed[:, :, None]) + speed[:, :, None] *
                          (q3[:, :, None] * b0 + 1.5 * q2[:, :, None] * b1 + .5 * q1[:, :, None] * b2),
                          m[:, :, None] * b0 + .5 * a[:, :, None] * b1))
        normalized = (values - self.center[:, None, :]) / self.width[:, None, :]
        jac = deriv / self.width[:, None, :, None]
        return np.concatenate(((1 + normalized).ravel(), (1 - normalized).ravel())), np.concatenate((jac.reshape(-1, 32), -jac.reshape(-1, 32)))


def main() -> None:
    """Solve one frozen attempt and save its diagnostics without altering prior evidence."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--grid-count', type=int, default=0)
    parser.add_argument('--refine', action='store_true')
    parser.add_argument('--maxiter', type=int, default=300)
    parser.add_argument('--tol', type=float, default=1e-7)
    parser.add_argument('--backend', choices=['ipopt', 'slsqp'], default='ipopt')
    parser.add_argument('--seed', choices=['native', 'restored', 'native40'], default='restored')
    args = parser.parse_args()
    out = Path(__file__).resolve().parent / 'results/native_reconstruction_20261106'
    p = PointwiseTiming(out, args.stride, args.grid_count, args.refine)
    seed = p.original if args.seed == 'native' else p.initial
    if args.seed == 'native40':
        seed = np.load(out / 'native_iteration_controls.npz')['40']
    x0 = seed / p.scale
    history = []
    start = time.perf_counter()
    if args.backend == 'ipopt':
        import cyipopt
        # Each cubic B-spline has compact support; only four control columns
        # affect a constraint at a given path sample.  Compute the pattern at
        # a positive point and use structural basis support (not numerical values).
        support = np.any(np.stack([np.abs(b) > 0 for b in p.b]), axis=0)
        pattern = np.broadcast_to(support[None, :, None, :], (4, len(support), 7, 32)).reshape(-1, 32)
        pattern = np.concatenate((pattern, pattern))
        rows, cols = np.nonzero(pattern)
        class Problem:
            def objective(self, x): return p.objective(x)
            def gradient(self, x): return p.gradient(x)
            def constraints(self, x): return p.values_jac(x)[0]
            def jacobianstructure(self): return rows, cols
            def jacobian(self, x): return p.values_jac(x)[1][rows, cols]
            def intermediate(self, *args): history.append(list(args[:8]))
        count = len(p.values_jac(x0)[0])
        problem = cyipopt.Problem(n=32, m=count, problem_obj=Problem(), lb=np.full(32, 1e-5),
                                 ub=np.full(32, 10000.), cl=np.zeros(count), cu=np.full(count, 1e19))
        problem.add_option('hessian_approximation', 'limited-memory')
        problem.add_option('max_iter', args.maxiter)
        problem.add_option('tol', args.tol)
        problem.add_option('print_level', 0)
        x, result = problem.solve(x0)
        status = int(result['status'])
        message = result['status_msg'].decode(errors='replace')
        multipliers = result['mult_g']
        stationarity = p.gradient(x) + p.values_jac(x)[1].T @ multipliers - result['mult_x_L'] + result['mult_x_U']
        result_details = {'status': status, 'message': message, 'stationarity_inf': float(np.max(np.abs(stationarity))),
                          'history_columns': ['algorithm_mode', 'iteration', 'objective', 'primal_infeasibility', 'dual_infeasibility', 'mu', 'step_norm', 'regularization']}
    else:
        def callback(x): history.append([p.objective(x), float(p.values_jac(x)[0].min())])
        result = minimize(p.objective, x0, jac=p.gradient, method='SLSQP', bounds=[(1e-5, 10000)] * 32,
                          constraints=[{'type': 'ineq', 'fun': lambda x: p.values_jac(x)[0],
                                        'jac': lambda x: p.values_jac(x)[1]}], callback=callback,
                          options={'maxiter': args.maxiter, 'ftol': 1e-9})
        x = result.x
        result_details = {'status': int(result.status), 'message': result.message,
                          'stationarity_inf': None, 'history_columns': ['objective', 'minimum_margin']}
    summary = {'backend': args.backend, 'seed': args.seed, 'stride': args.stride, 'tolerance': args.tol,
               'scaled_control_lower_bound': 1e-5, 'scaled_control_upper_bound': 10000.,
               'collocation_points': len(p.b[0]), 'pointwise_inequalities': len(p.values_jac(x)[0]),
               'duration_s': p.objective(x), 'initial_duration_s': p.objective(x0),
               'min_margin': float(p.values_jac(x)[0].min()), 'elapsed_s': time.perf_counter() - start,
               'iterations_recorded': len(history), **result_details}
    tag = f'pointwise_{args.backend}_{args.seed}_stride{args.stride}_grid{args.grid_count}_refine{int(args.refine)}_tol{args.tol:g}_upper10000'
    np.savez_compressed(out / (tag + '.npz'), controls=x*p.scale, history=np.array(history), grid=p.grid)
    (out / (tag + '.json')).write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
