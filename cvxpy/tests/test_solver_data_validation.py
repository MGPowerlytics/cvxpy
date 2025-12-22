
"""
Copyright, the CVXPY authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import numpy as np
import pytest

import cvxpy as cp
from cvxpy.tests.base_test import BaseTest


class TestSolverDataValidation(BaseTest):
    """Test that NaN/Inf values in problem data are caught early.

    These tests verify that NaN/Inf values in constants are detected
    during apply_parameters() before being sent to the solver.
    Note: Parameters are already validated at assignment time, so
    NaN/Inf in parameters are caught earlier.
    """

    def test_inf_constant_in_constraint(self):
        """Inf in a constraint constant should raise ValueError during solve."""
        x = cp.Variable()
        c = cp.Constant(np.inf)
        prob = cp.Problem(cp.Minimize(x), [x >= c])

        if cp.SCS in cp.installed_solvers():
            with pytest.raises(ValueError, match="contains NaN or Inf"):
                prob.solve(solver=cp.SCS)

    def test_nan_constant_in_objective(self):
        """NaN in objective coefficients should raise ValueError."""
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(x * cp.Constant(np.nan)), [x >= 0])

        if cp.SCS in cp.installed_solvers():
            with pytest.raises(ValueError, match="contains NaN or Inf"):
                prob.solve(solver=cp.SCS)

    def test_inf_constant_in_objective(self):
        """Inf in objective coefficients should raise ValueError."""
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(x + cp.Constant(np.inf)), [x >= 0])

        if cp.SCS in cp.installed_solvers():
            with pytest.raises(ValueError, match="contains NaN or Inf"):
                prob.solve(solver=cp.SCS)

    def test_valid_problem_solves(self):
        """A valid problem should still solve correctly."""
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(x), [x >= 1])

        if cp.SCS in cp.installed_solvers():
            prob.solve(solver=cp.SCS)
            self.assertAlmostEqual(x.value, 1.0, places=3)

    def test_qp_solver_with_inf(self):
        """Test that QP solvers also catch Inf in problem data."""
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(x**2), [x >= cp.Constant(np.inf)])

        if cp.OSQP in cp.installed_solvers():
            with pytest.raises(ValueError, match="contains NaN or Inf"):
                prob.solve(solver=cp.OSQP)

    def test_ignore_nan_option(self):
        """Test that ignore_nan=True skips validation."""
        x = cp.Variable()
        # Create a problem with inf in constraint
        prob = cp.Problem(cp.Minimize(x), [x >= cp.Constant(np.inf)])

        if cp.SCS in cp.installed_solvers():
            # With ignore_nan=True, no ValueError should be raised during
            # apply_parameters. The solver may still fail or return
            # a non-optimal status, but that's expected behavior.
            try:
                prob.solve(solver=cp.SCS, ignore_nan=True)
                # The solver may return inf, nan, or fail gracefully
                # We just verify no ValueError was raised by our validation
            except ValueError as e:
                if "contains NaN or Inf" in str(e):
                    pytest.fail("ignore_nan=True should skip validation")
                # Other ValueErrors from the solver itself are acceptable
            except Exception:
                # Solver-level exceptions are acceptable when data contains inf
                pass

    def test_neg_inf_in_constraint(self):
        """Negative infinity in constraint should raise ValueError."""
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(x), [x <= cp.Constant(-np.inf)])

        if cp.SCS in cp.installed_solvers():
            with pytest.raises(ValueError, match="contains NaN or Inf"):
                prob.solve(solver=cp.SCS)

    def test_nan_in_matrix_constraint(self):
        """NaN in matrix constraint data should raise ValueError."""
        X = cp.Variable((2, 2))
        A = np.array([[1, np.nan], [0, 1]])
        prob = cp.Problem(cp.Minimize(cp.sum(X)), [X == A])

        if cp.SCS in cp.installed_solvers():
            with pytest.raises(ValueError, match="contains NaN or Inf"):
                prob.solve(solver=cp.SCS)

    def test_inf_in_quadratic_objective(self):
        """Inf coefficient in quadratic objective should raise ValueError."""
        x = cp.Variable()
        # Create quadratic form with inf coefficient
        prob = cp.Problem(cp.Minimize(cp.Constant(np.inf) * x**2), [x >= 0])

        if cp.OSQP in cp.installed_solvers():
            with pytest.raises(ValueError, match="contains NaN or Inf"):
                prob.solve(solver=cp.OSQP)

    def test_inf_from_expression_overflow(self):
        """Inf resulting from expression overflow should raise ValueError."""
        x = cp.Variable()
        # Very large values that will overflow to inf when combined
        large_val = 1e308
        prob = cp.Problem(cp.Minimize(x), [x >= large_val * 2])

        if cp.SCS in cp.installed_solvers():
            with pytest.raises(ValueError, match="contains NaN or Inf"):
                prob.solve(solver=cp.SCS)

    def test_nan_in_socp_constraint(self):
        """NaN in second-order cone constraint should raise ValueError."""
        x = cp.Variable(2)
        t = cp.Variable()
        A = np.array([[1, np.nan], [0, 1]])
        prob = cp.Problem(cp.Minimize(t), [cp.norm(A @ x) <= t])

        if cp.SCS in cp.installed_solvers():
            with pytest.raises(ValueError, match="contains NaN or Inf"):
                prob.solve(solver=cp.SCS)

    def test_inf_in_equality_constraint(self):
        """Inf in equality constraint should raise ValueError."""
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(x**2), [x == cp.Constant(np.inf)])

        if cp.SCS in cp.installed_solvers():
            with pytest.raises(ValueError, match="contains NaN or Inf"):
                prob.solve(solver=cp.SCS)

    def test_multiple_solves_with_and_without_ignore_nan(self):
        """Test that ignore_nan doesn't persist across solves."""
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(x), [x >= cp.Constant(np.inf)])

        if cp.SCS in cp.installed_solvers():
            # First solve with ignore_nan=True - should not raise our error
            try:
                prob.solve(solver=cp.SCS, ignore_nan=True)
            except Exception:
                pass  # Solver errors are expected

            # Second solve without ignore_nan - SHOULD raise our error
            with pytest.raises(ValueError, match="contains NaN or Inf"):
                prob.solve(solver=cp.SCS)

    def test_clarabel_with_inf(self):
        """Test that Clarabel also catches Inf in problem data."""
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(x), [x >= cp.Constant(np.inf)])

        if cp.CLARABEL in cp.installed_solvers():
            with pytest.raises(ValueError, match="contains NaN or Inf"):
                prob.solve(solver=cp.CLARABEL)
