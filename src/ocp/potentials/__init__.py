"""Convex potentials with closed-form gradients."""

from .base import Potential
from .logsumexp import LogSumExpPotential

__all__ = ["Potential", "LogSumExpPotential"]

