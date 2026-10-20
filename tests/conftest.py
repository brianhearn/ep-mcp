"""Shared pytest options (pytest only honours pytest_addoption in conftest.py)."""

from __future__ import annotations


def pytest_addoption(parser):
    parser.addoption("--pack", action="store", default=None, help="Path to an ExpertPack (integration tests)")
    parser.addoption("--golden", action="store", default=None,
                     help="Golden eval YAML file or directory (also EP_GOLDEN env var)")
    parser.addoption("--packs-root", action="store", default=None,
                     help="Directory containing packs by slug (also EP_PACKS_ROOT env var)")
