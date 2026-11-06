# SPDX-License-Identifier: Apache-2.0
"""Optional IPOPT backend for the joint free-time nonlinear program.

Requires separately installed cyipopt + IPOPT libraries. Uses the same CUDA
autograd Jacobians and physical constraints as the SLSQP pilot. Provides
local KKT solutions only, not a global optimality certificate.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np


def solve_ipopt(oracle, x0: np.ndarray, bounds):
    try:
        import cyipopt
    except ImportError as exc:
        raise RuntimeError(
            "cyipopt + IPOPT are required for solver='ipopt'; install them "
            "in a separate research environment or use --solver slsqp"
        ) from exc

    n = len(x0)
    m = len(oracle._evaluate(x0,False))
    lb,ub = (np.asarray([b[i] for b in bounds],dtype=np.float64) for i in (0,1))
    nonzero_i,nonzero_j = np.indices((m,n),dtype=np.int32)
    class NLP:
        n_obj=0
        n_con=0

        def objective(self,z):
            self.n_obj+=1
            return float(np.exp(z[-1])+1000*z[-2])

        def gradient(self,z):
            return np.r_[np.zeros(n-2),1000.,np.exp(z[-1])]

        def constraints(self,z):
            self.n_con+=1
            return oracle._evaluate(z,False)

        def jacobian(self,z):
            return oracle._evaluate(z,True).ravel()

        def jacobianstructure(self):
            return nonzero_i.ravel(),nonzero_j.ravel()

    nlp=NLP()
    problem=cyipopt.Problem(n=n,m=m,problem_obj=nlp,
                           lb=lb,ub=ub,cl=np.zeros(m),cu=np.full(m,1e19))
    problem.add_option("print_level",0)
    problem.add_option("sb","yes")
    problem.add_option("max_iter",int(oracle.cfg.maxiter))
    problem.add_option("tol",float(oracle.cfg.ftol))
    problem.add_option("acceptable_tol",max(1e-6,float(oracle.cfg.ftol)*10))
    problem.add_option("hessian_approximation","limited-memory")
    problem.add_option("mu_strategy","adaptive")
    x,info=problem.solve(np.asarray(x0,dtype=np.float64))
    status=int(info.get("status",999))
    message=info.get("status_msg","")
    if isinstance(message,bytes):
        message=message.decode(errors="replace")
    return SimpleNamespace(x=np.asarray(x),success=status in (0,1),
                           nit=int(info.get("iter_count",0)),nfev=nlp.n_obj,
                           message=f"IPOPT {status}: {message}")
