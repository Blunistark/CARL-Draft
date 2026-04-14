"""PID controller baseline for comparison with DRL agents."""
from __future__ import annotations

from typing import Any


class PIDBaseline:
    """Single-axis PID controller used as a DRL performance baseline.

    Implements a discrete-time PID update rule:
        u(t) = Kp * e(t) + Ki * sum(e) * dt + Kd * (e(t) - e(t-1)) / dt

    Args:
        kp: Proportional gain.
        ki: Integral gain.
        kd: Derivative gain.
        dt: Control timestep in seconds.
        output_limits: Optional ``(min, max)`` tuple to clamp controller output.
    """

    def __init__(
        self,
        kp: float = 1.0,
        ki: float = 0.0,
        kd: float = 0.0,
        dt: float = 0.05,
        output_limits: tuple[float, float] | None = None,
    ) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.output_limits = output_limits

        self._integral: float = 0.0
        self._prev_error: float = 0.0

    def reset(self) -> None:
        """Reset integrator and derivative state."""
        self._integral = 0.0
        self._prev_error = 0.0

    def compute(self, setpoint: float, measurement: float) -> float:
        """Compute PID output for a single control step.

        Args:
            setpoint: Desired value (reference signal).
            measurement: Current measured value.

        Returns:
            Controller output, optionally clamped to ``output_limits``.
        """
        error = setpoint - measurement
        self._integral += error * self.dt
        derivative = (error - self._prev_error) / self.dt
        self._prev_error = error

        output = self.kp * error + self.ki * self._integral + self.kd * derivative

        if self.output_limits is not None:
            lo, hi = self.output_limits
            output = max(lo, min(hi, output))

        return output

    def select_action(self, state: Any) -> Any:
        """Compatibility shim matching the :class:`~carl.agents.BaseAgent` interface.

        Subclasses should override this with domain-specific setpoint extraction.

        Args:
            state: Environment observation.

        Returns:
            Controller output (scalar or array depending on subclass).
        """
        raise NotImplementedError(
            "select_action must be implemented by a concrete PID subclass."
        )
