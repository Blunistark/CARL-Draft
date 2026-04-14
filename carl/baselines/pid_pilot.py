"""Full PID autopilot for fixed-wing aircraft with pitch, roll, and yaw control."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from carl.baselines.pid_baseline import PIDBaseline


@dataclass
class PIDGains:
    """Container for PID gain tuples per control axis.

    Each tuple holds ``(kp, ki, kd)`` for the respective axis.
    """

    pitch: tuple[float, float, float] = (1.0, 0.0, 0.1)
    roll: tuple[float, float, float] = (1.0, 0.0, 0.1)
    yaw: tuple[float, float, float] = (0.5, 0.0, 0.05)
    throttle: tuple[float, float, float] = (1.0, 0.1, 0.0)


class PIDPilot:
    """Multi-axis PID autopilot for fixed-wing aircraft control.

    Orchestrates four independent PID controllers (pitch, roll, yaw,
    throttle) to track a reference trajectory.  Each axis can be tuned
    independently via :class:`PIDGains`.

    Args:
        gains: :class:`PIDGains` instance with per-axis ``(kp, ki, kd)`` tuples.
        dt: Control timestep in seconds.
        action_limits: Global ``(min, max)`` clamp applied to all control outputs.
    """

    def __init__(
        self,
        gains: PIDGains | None = None,
        dt: float = 0.05,
        action_limits: tuple[float, float] = (-1.0, 1.0),
    ) -> None:
        gains = gains or PIDGains()
        self.dt = dt
        self.action_limits = action_limits

        self._pitch_pid = PIDBaseline(*gains.pitch, dt=dt, output_limits=action_limits)
        self._roll_pid = PIDBaseline(*gains.roll, dt=dt, output_limits=action_limits)
        self._yaw_pid = PIDBaseline(*gains.yaw, dt=dt, output_limits=action_limits)
        self._throttle_pid = PIDBaseline(
            *gains.throttle, dt=dt, output_limits=(0.0, 1.0)
        )

    def reset(self) -> None:
        """Reset all axis PID controllers."""
        for pid in (
            self._pitch_pid,
            self._roll_pid,
            self._yaw_pid,
            self._throttle_pid,
        ):
            pid.reset()

    def compute(
        self,
        pitch_ref: float,
        roll_ref: float,
        yaw_ref: float,
        airspeed_ref: float,
        pitch: float,
        roll: float,
        yaw: float,
        airspeed: float,
    ) -> dict[str, float]:
        """Compute control surface deflections and throttle command.

        Args:
            pitch_ref: Target pitch angle (radians).
            roll_ref: Target roll angle (radians).
            yaw_ref: Target yaw angle (radians).
            airspeed_ref: Target airspeed (m/s).
            pitch: Current pitch angle (radians).
            roll: Current roll angle (radians).
            yaw: Current yaw angle (radians).
            airspeed: Current airspeed (m/s).

        Returns:
            Dict with keys ``elevator``, ``aileron``, ``rudder``, ``throttle``.
        """
        return {
            "elevator": self._pitch_pid.compute(pitch_ref, pitch),
            "aileron": self._roll_pid.compute(roll_ref, roll),
            "rudder": self._yaw_pid.compute(yaw_ref, yaw),
            "throttle": self._throttle_pid.compute(airspeed_ref, airspeed),
        }

    def select_action(self, state: Any) -> Any:
        """Select control action from the full state vector.

        Subclasses or wrappers should unpack ``state`` into the individual
        reference / measurement pairs before calling :meth:`compute`.

        Args:
            state: Environment observation containing references and measurements.

        Returns:
            Control action array or dict.
        """
        raise NotImplementedError(
            "select_action must be implemented by a concrete PIDPilot subclass."
        )
