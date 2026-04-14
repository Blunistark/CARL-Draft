"""Smoke tests: verify that the carl package and submodules import correctly."""
import importlib

import pytest

MODULES = [
    "carl",
    "carl.agents",
    "carl.agents.base_agent",
    "carl.agents.baselines",
    "carl.agents.pioneer",
    "carl.baselines",
    "carl.curriculum",
    "carl.evaluation",
    "carl.logging",
    "carl.models",
    "carl.registry",
    "carl.reward",
    "carl.training",
    "carl.utils",
    "carl.viz",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_import(module_name):
    mod = importlib.import_module(module_name)
    assert mod is not None
