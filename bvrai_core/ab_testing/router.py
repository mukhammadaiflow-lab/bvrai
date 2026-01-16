"""Variant routing for A/B testing."""

import hashlib
import logging
import random
from typing import Dict, List, Optional, Tuple

from .models import Experiment, ExperimentVariant

logger = logging.getLogger(__name__)


class VariantRouter:
    """
    Routes calls to experiment variants using various assignment strategies.

    Supports:
    - Random assignment (default)
    - Weighted random based on traffic percentages
    - Deterministic assignment based on phone number (sticky)
    - Multi-armed bandit (adaptive allocation)
    """

    def __init__(self):
        # Cache for sticky assignments
        self._sticky_assignments: Dict[str, Dict[str, str]] = {}  # exp_id -> phone -> variant_id

    def select_variant(
        self,
        experiment: Experiment,
        phone_number: str = "",
        sticky: bool = True,
    ) -> Optional[ExperimentVariant]:
        """
        Select a variant for a call.

        Args:
            experiment: The experiment
            phone_number: Phone number (for sticky assignment)
            sticky: Whether to use deterministic assignment

        Returns:
            Selected variant or None if experiment is not active
        """
        if not experiment.is_active:
            return None

        if not experiment.variants:
            return None

        # Check traffic percentage (should this call be in experiment)
        if experiment.traffic_percentage < 100.0:
            if random.random() * 100 > experiment.traffic_percentage:
                return None  # Call not included in experiment

        # Check for existing sticky assignment
        if sticky and phone_number:
            existing = self._get_sticky_assignment(experiment.id, phone_number)
            if existing:
                for variant in experiment.variants:
                    if variant.id == existing:
                        return variant

        # Select variant based on traffic percentages
        variant = self._weighted_random_selection(experiment.variants)

        # Store sticky assignment
        if sticky and phone_number and variant:
            self._set_sticky_assignment(experiment.id, phone_number, variant.id)

        return variant

    def select_variant_deterministic(
        self,
        experiment: Experiment,
        identifier: str,
    ) -> Optional[ExperimentVariant]:
        """
        Deterministically select a variant based on an identifier.

        This ensures the same identifier always gets the same variant.
        """
        if not experiment.is_active or not experiment.variants:
            return None

        # Hash the experiment ID + identifier
        hash_input = f"{experiment.id}:{identifier}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)

        # Map to variant based on traffic percentages
        cumulative = 0.0
        normalized_hash = (hash_value % 10000) / 100.0  # 0-100

        for variant in experiment.variants:
            cumulative += variant.traffic_percentage
            if normalized_hash < cumulative:
                return variant

        # Fallback to last variant
        return experiment.variants[-1]

    def get_variant_for_call(
        self,
        experiments: List[Experiment],
        phone_number: str = "",
        agent_id: str = "",
    ) -> List[Tuple[Experiment, ExperimentVariant]]:
        """
        Get variants for all active experiments applicable to a call.

        Returns list of (experiment, variant) tuples.
        """
        assignments = []

        for experiment in experiments:
            # Check if experiment applies to this agent
            if experiment.agent_ids and agent_id not in experiment.agent_ids:
                continue

            variant = self.select_variant(experiment, phone_number)
            if variant:
                assignments.append((experiment, variant))

        return assignments

    def _weighted_random_selection(
        self,
        variants: List[ExperimentVariant],
    ) -> Optional[ExperimentVariant]:
        """Select a variant using weighted random selection."""
        if not variants:
            return None

        total_weight = sum(v.traffic_percentage for v in variants)
        if total_weight <= 0:
            return random.choice(variants)

        r = random.random() * total_weight
        cumulative = 0.0

        for variant in variants:
            cumulative += variant.traffic_percentage
            if r <= cumulative:
                return variant

        return variants[-1]

    def _get_sticky_assignment(
        self,
        experiment_id: str,
        phone_number: str,
    ) -> Optional[str]:
        """Get sticky assignment for a phone number."""
        exp_assignments = self._sticky_assignments.get(experiment_id, {})
        return exp_assignments.get(phone_number)

    def _set_sticky_assignment(
        self,
        experiment_id: str,
        phone_number: str,
        variant_id: str,
    ) -> None:
        """Set sticky assignment for a phone number."""
        if experiment_id not in self._sticky_assignments:
            self._sticky_assignments[experiment_id] = {}
        self._sticky_assignments[experiment_id][phone_number] = variant_id

    def clear_sticky_assignments(self, experiment_id: str) -> None:
        """Clear sticky assignments for an experiment."""
        if experiment_id in self._sticky_assignments:
            del self._sticky_assignments[experiment_id]


class BanditRouter(VariantRouter):
    """
    Multi-armed bandit variant router for adaptive allocation.

    Uses Thompson Sampling to automatically allocate more traffic
    to better-performing variants.
    """

    def __init__(self, exploration_weight: float = 1.0):
        super().__init__()
        self.exploration_weight = exploration_weight

        # Success/failure counts for Thompson Sampling
        self._successes: Dict[str, Dict[str, int]] = {}  # exp_id -> variant_id -> count
        self._failures: Dict[str, Dict[str, int]] = {}

    def select_variant_adaptive(
        self,
        experiment: Experiment,
        phone_number: str = "",
    ) -> Optional[ExperimentVariant]:
        """
        Select variant using Thompson Sampling.

        Balances exploration vs exploitation based on observed performance.
        """
        if not experiment.is_active or not experiment.variants:
            return None

        # Check for sticky assignment first
        if phone_number:
            existing = self._get_sticky_assignment(experiment.id, phone_number)
            if existing:
                for variant in experiment.variants:
                    if variant.id == existing:
                        return variant

        # Get or initialize counts
        exp_successes = self._successes.get(experiment.id, {})
        exp_failures = self._failures.get(experiment.id, {})

        # Sample from beta distribution for each variant
        samples = []
        for variant in experiment.variants:
            successes = exp_successes.get(variant.id, 1)
            failures = exp_failures.get(variant.id, 1)

            # Thompson Sampling: sample from Beta(successes, failures)
            import random as rnd
            sample = rnd.betavariate(
                successes * self.exploration_weight,
                failures * self.exploration_weight,
            )
            samples.append((sample, variant))

        # Select variant with highest sample
        _, selected = max(samples, key=lambda x: x[0])

        # Store sticky assignment
        if phone_number:
            self._set_sticky_assignment(experiment.id, phone_number, selected.id)

        return selected

    def record_outcome(
        self,
        experiment_id: str,
        variant_id: str,
        success: bool,
    ) -> None:
        """Record outcome for bandit learning."""
        if experiment_id not in self._successes:
            self._successes[experiment_id] = {}
            self._failures[experiment_id] = {}

        if success:
            self._successes[experiment_id][variant_id] = (
                self._successes[experiment_id].get(variant_id, 0) + 1
            )
        else:
            self._failures[experiment_id][variant_id] = (
                self._failures[experiment_id].get(variant_id, 0) + 1
            )

    def get_variant_probabilities(
        self,
        experiment: Experiment,
    ) -> Dict[str, float]:
        """
        Get current probability estimates for each variant.

        Returns estimated success probability based on observed data.
        """
        exp_successes = self._successes.get(experiment.id, {})
        exp_failures = self._failures.get(experiment.id, {})

        probabilities = {}
        for variant in experiment.variants:
            successes = exp_successes.get(variant.id, 1)
            failures = exp_failures.get(variant.id, 1)

            # Beta distribution mean
            prob = successes / (successes + failures)
            probabilities[variant.id] = prob

        return probabilities
