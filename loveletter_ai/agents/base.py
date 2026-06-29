"""Shared agent protocol."""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Protocol

from loveletter_ai.actions import Action
from loveletter_ai.state import Observation


class Agent(Protocol):
    """Minimal policy interface used by environment runners."""

    name: str

    def choose_action(
        self,
        observation: Observation,
        legal_actions: Sequence[Action],
        rng: random.Random,
    ) -> Action:
        """Choose one legal action from an environment-provided action list."""
