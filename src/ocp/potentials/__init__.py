"""Convex potentials with closed-form gradients."""

from .base import Potential
from .logsumexp import LogSumExpPotential
from .moreau_max import MoreauMaxPotential

__all__ = ["Potential", "LogSumExpPotential", "MoreauMaxPotential"]

