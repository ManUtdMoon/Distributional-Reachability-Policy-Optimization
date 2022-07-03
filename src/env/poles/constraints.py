""" A helper class for state constraints, 
borrow from 
https://github.com/utiasDSL/safe-control-gym/blob/main/safe_control_gym/envs/constraints.py
"""

from enum import Enum

import numpy as np

class ConstrainedVariableType(str, Enum):
    """Allowable constraint type specifiers.

    """

    STATE = "state"  # Constraints who are a function of the state X.


class Constraint:
    """Implements a (state-wise/trajectory-wise/stateful) constraint.

    A constraint can contain multiple scalar-valued constraint functions.
    Each should be represented as g(x) <= 0.

    Attributes:
        constrained_variable: the variable(s) from env to be constrained.
        dim (int): Total number of input dimensions to be constrained, i.e. dim of x. 
        num_constraints (int): total number of output dimensions or number of constraints, i.e. dim of g(x).
        sym_func (Callable): the symbolic function of the constraint, can take in np.array or CasADi variable.
        
    """
    
    def __init__(self,
                 state_dim,
                 constrained_variable: ConstrainedVariableType,
                 strict: bool=False,
                 active_dims=None,
                 tolerance=None,
                 **kwargs
                 ):
        """Defines params (e.g. bounds) and state.

        Args:
            env (safe_control_gym.envs.bechmark_env.BenchmarkEnv): The environment the constraint is for.
            constrained_variable (ConstrainedVariableType): Specifies the input type to the constraint as a constraint
                                                         that acts on the state, input, or both.
            strict (optional, bool): Whether the constraint is violated also when equal to its threshold.
            active_dims (list of ints): Filters the constraint to only act only select certian dimensions.
            tolerance (list or np.array): The distance from the constraint at which is_almost_active returns True.

        """
        self.constrained_variable = ConstrainedVariableType(constrained_variable)
        if self.constrained_variable == ConstrainedVariableType.STATE:
            self.dim = state_dim
        else:
            raise NotImplementedError('[ERROR] invalid constrained_variable (use STATE).')
        # Save the strictness attribute
        self.strict = strict
        # Only want to select specific dimensions, implemented via a filter matrix.
        if active_dims is not None:
            if isinstance(active_dims, int):
                active_dims = [active_dims]
            assert isinstance(active_dims, (list, np.ndarray)), '[ERROR] active_dims is not a list/array.'
            assert (len(active_dims) <= self.dim), '[ERROR] more active_dim than constrainable self.dim'
            assert all(isinstance(n, int) for n in active_dims), '[ERROR] non-integer active_dim.'
            assert all((n < self.dim) for n in active_dims), '[ERROR] active_dim not stricly smaller than self.dim.'
            assert (len(active_dims) == len(set(active_dims))), '[ERROR] duplicates in active_dim'
            self.constraint_filter = np.eye(self.dim)[active_dims]
            self.dim = len(active_dims)
        else:
            self.constraint_filter = np.eye(self.dim)
        if tolerance is not None:
            self.tolerance = np.array(tolerance, ndmin=1)
        else:
            self.tolerance = None

    def get_symbolic_model(self,
                           env
                           ):
        """Gets the symbolic form of the constraint function.

        Args:
            env: The environment to constrain.

        Returns:
            obj: The symbolic form of the constraint.

        """
        return NotImplementedError

    def get_value(self,
                  states: np.array
                  ):
        """Gets the constraint function value.

        Args:
            states (n, d): The environment to constrain.

        Returns:
            ndarray (n, con_num): The evaulation of the constraint.

        """
        if len(states.shape) == 1:
            states = states[np.newaxis, ...]
        assert len(states.shape) >= 2

        return self.sym_func(states)

    def is_violated(self,
                    states: np.array,
                    c_value=None
                    ):
        """Checks if constraint is violated.

        Args:
            states (n, d): The environment to constrain.
            c_value: an already calculated constraint value (no need to recompute).

        Returns:
            bool (n,) : Whether the constraint was violeted.

        """
        if len(states.shape) == 1:
            states = states[np.newaxis, ...]
        assert len(states.shape) >= 2

        if c_value is None:
            c_value = self.get_value(states)
        if self.strict:
            flag = np.any(np.greater_equal(c_value, 0.), axis=-1)
        else:
            flag = np.any(np.greater(c_value, 0.), axis=-1)
        return flag

    def is_almost_active(self,
                         states: np.array,
                         c_value=None
                         ):
        """Checks if constraint is nearly violated.

        This is checked by using a slack variable (from init args).
        This can be used for reward shaping/constraint penalty in RL methods.

        """
        if len(states.shape) == 1:
            states = states[np.newaxis, ...]
        assert len(states.shape) >= 2

        if not hasattr(self, "tolerance") or self.tolerance is None:
            assert 0
        if c_value is None:
            c_value = self.get_value(states)
        flag = np.any(np.greater(c_value + self.tolerance, 0.), axis=-1)
        return flag

    def check_tolerance_shape(self):
        if self.tolerance is not None and len(self.tolerance) != self.num_constraints:
            raise ValueError('[ERROR] the tolerance dimension does not match the number of constraints.')

class LinearConstraint(Constraint):
    """Constraint class for constraints of the form A @ x <= b.

    """

    def __init__(self,
                 state_dim,
                 A: np.ndarray,
                 b: np.ndarray,
                 constrained_variable: ConstrainedVariableType,
                 strict: bool=False,
                 active_dims=None,
                 tolerance=None
                 ):
        """Initialize the class.

        Args:
            env (BenchmarkEnv): The environment to constraint.
            A (np.array or list): A matrix of the constraint (self.num_constraints by self.dim).
            b (np.array or list): b matrix of the constraint (1D array self.num_constraints)
                                  constrained_variable (ConstrainedVariableType): Type of constraint.
            strict (optional, bool): Whether the constraint is violated also when equal to its threshold.
            active_dims (list or int): List specifying which dimensions the constraint is active for.
            tolerance (float): The distance at which is_almost_active(env) triggers.

        """
        super().__init__(state_dim, constrained_variable, strict=strict, active_dims=active_dims, tolerance=tolerance)
        A = np.array(A, ndmin=1)
        b = np.array(b, ndmin=1)
        assert A.shape[1] == self.dim, '[ERROR] A has the wrong dimension!'
        self.A = A
        assert b.shape[0] == A.shape[0], '[ERROR] Dimension 0 of b does not match A!'
        self.b = b
        self.num_constraints = A.shape[0]
        
        """Matmul shape explanation
        active_dim: (a,)
        con_dim: (c,); 
        A: (c, a), b: (c,)
        x: (*, d)
        constraint_filter: (a, d)

        therefore, x @ filter.T @ A.T = (*,d)@(d,a)@(a,c) = (*,c)
        """
        self.sym_func = lambda x: \
            x @ self.constraint_filter.transpose() @ self.A.transpose() - self.b
        self.check_tolerance_shape()

    def get_symbolic_model(self):
        """Gets the symbolic form of the constraint function.

        Returns:
            lambda: The symbolic form of the constraint.

        """
        return self.sym_func

class BoundedConstraint(LinearConstraint):
    """ Class for bounded constraints lb <= x <= ub as polytopic constraints -Ix + lb <= 0 and Ix - ub <= 0.

    """

    def __init__(self,
                 state_dim,
                 lower_bounds: np.ndarray,
                 upper_bounds: np.ndarray,
                 constrained_variable: ConstrainedVariableType,
                 strict: bool=False,
                 active_dims=None,
                 tolerance=None):
        """Initialize the constraint.

        Args:
            env (BenchmarkEnv): The environment to constraint.
            lower_bounds (np.array or list): Lower bound of constraint.
            upper_bounds (np.array or list): Uppbound of constraint.
            constrained_variable (ConstrainedVariableType): Type of constraint.
            strict (optional, bool): Whether the constraint is violated also when equal to its threshold.
            active_dims (list or int): List specifying which dimensions the constraint is active for.
            tolerance (float): The distance at which is_almost_active(env) triggers.

        """
        self.lower_bounds = np.array(lower_bounds, ndmin=1)
        self.upper_bounds = np.array(upper_bounds, ndmin=1)
        dim = self.lower_bounds.shape[0]
        A = np.vstack((-np.eye(dim), np.eye(dim)))
        b = np.hstack((-self.lower_bounds, self.upper_bounds))
        super().__init__(state_dim, A, b, constrained_variable, strict=strict, active_dims=active_dims, tolerance=tolerance)
        self.check_tolerance_shape()