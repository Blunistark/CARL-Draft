"""Registry for Pilot Operating Handbook (POH) aircraft configurations."""
from __future__ import annotations

from typing import Any


class POHRegistry:
    """Registry for Pilot Operating Handbook (POH) configurations and aircraft performance data.

    Stores named aircraft configurations (aerodynamic coefficients, performance
    envelopes, actuator limits) that can be looked up by agents and environments
    at runtime.

    Args:
        strict: If ``True``, raise ``KeyError`` on duplicate ``register`` calls
            instead of silently overwriting.
    """

    def __init__(self, strict: bool = False) -> None:
        self.strict = strict
        self._store: dict[str, dict[str, Any]] = {}

    def register(self, name: str, config: dict[str, Any]) -> None:
        """Register a named aircraft configuration.

        Args:
            name: Unique identifier for the aircraft / POH entry.
            config: Dict of aircraft parameters (mass, wingspan, Cl_alpha, etc.).

        Raises:
            KeyError: If ``strict=True`` and ``name`` is already registered.
        """
        if self.strict and name in self._store:
            raise KeyError(f"'{name}' is already registered in POHRegistry.")
        self._store[name] = config

    def get(self, name: str) -> dict[str, Any]:
        """Retrieve a registered configuration by name.

        Args:
            name: The identifier to look up.

        Returns:
            The configuration dict associated with ``name``.

        Raises:
            KeyError: If ``name`` has not been registered.
        """
        if name not in self._store:
            raise KeyError(
                f"'{name}' not found in POHRegistry. "
                f"Available entries: {list(self._store)}"
            )
        return self._store[name]

    def list_all(self) -> list[str]:
        """Return a sorted list of all registered configuration names.

        Returns:
            Sorted list of registered name strings.
        """
        return sorted(self._store)

    def __contains__(self, name: str) -> bool:
        return name in self._store

    def __len__(self) -> int:
        return len(self._store)
