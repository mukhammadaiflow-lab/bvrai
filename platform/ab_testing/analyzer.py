"""Statistical analysis for A/B testing experiments."""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

from .models import (
    Experiment,
    ExperimentVariant,
    ExperimentResult,
    VariantMetricResult,
    MetricComparison,
    MetricType,
    StatisticalSignificance,
    ExperimentStatus,
)

logger = logging.getLogger(__name__)


@dataclass
class SampleStatistics:
    """Statistics for a sample."""
    n: int
    mean: float
    variance: float
    std_dev: float


class ExperimentAnalyzer:
    """
    Statistical analyzer for A/B testing experiments.

    Provides:
    - Statistical significance testing
    - Confidence intervals
    - Effect size calculations
    - Winner determination
    """

    def analyze_experiment(self, experiment: Experiment) -> ExperimentResult:
        """
        Perform full statistical analysis of an experiment.

        Returns ExperimentResult with all metrics analyzed.
        """
        result = ExperimentResult(
            experiment_id=experiment.id,
            status=experiment.status,
            start_time=experiment.started_at,
            end_time=experiment.completed_at,
            total_participants=experiment.total_participants,
        )

        control = experiment.control_variant
        if not control:
            result.recommendation = "No control variant defined"
            return result

        # Analyze each metric
        for metric in experiment.metrics:
            # Get values for all variants
            variant_results = []
            for variant in experiment.variants:
                metric_result = self._calculate_variant_metric(variant, metric.metric_type)
                variant_results.append(metric_result)

            result.variant_results.extend(variant_results)

            # Compare each treatment to control
            control_result = next(
                (r for r in variant_results if r.variant_id == control.id),
                None
            )

            if not control_result:
                continue

            for treatment in experiment.treatment_variants:
                treatment_result = next(
                    (r for r in variant_results if r.variant_id == treatment.id),
                    None
                )

                if treatment_result:
                    comparison = self._compare_variants(
                        control_result,
                        treatment_result,
                        metric.name,
                        experiment.confidence_level,
                    )
                    result.metric_comparisons.append(comparison)

        # Determine winner based on primary metric
        primary_metric = next(
            (m for m in experiment.metrics if m.is_primary),
            None
        )

        if primary_metric:
            winner = self._determine_winner(
                experiment,
                primary_metric.metric_type,
                experiment.confidence_level,
            )
            if winner:
                result.winning_variant_id = winner.id
                result.winning_variant_name = winner.name

        # Generate recommendation
        result.recommendation = self._generate_recommendation(experiment, result)

        return result

    def _calculate_variant_metric(
        self,
        variant: ExperimentVariant,
        metric_type: MetricType,
    ) -> VariantMetricResult:
        """Calculate metric value and statistics for a variant."""
        value = 0.0
        n = variant.total_calls

        if metric_type == MetricType.SUCCESS_RATE:
            value = variant.success_rate
        elif metric_type == MetricType.AVG_DURATION:
            value = variant.avg_duration
        elif metric_type == MetricType.SENTIMENT_SCORE:
            value = variant.avg_sentiment
        elif metric_type == MetricType.CONVERSION_RATE:
            value = variant.conversion_rate
        elif metric_type == MetricType.CALL_COMPLETION_RATE:
            value = variant.success_rate
        else:
            # Custom metric
            value = variant.custom_metrics.get(metric_type.value, 0.0)

        # Calculate confidence interval for proportion metrics
        ci_lower, ci_upper, se = self._calculate_confidence_interval(
            value / 100.0 if value > 1 else value,  # Convert percentage to proportion
            n,
            0.95,
        )

        return VariantMetricResult(
            variant_id=variant.id,
            variant_name=variant.name,
            metric_name=metric_type.value,
            value=value,
            sample_size=n,
            confidence_interval_lower=ci_lower * 100 if value > 1 else ci_lower,
            confidence_interval_upper=ci_upper * 100 if value > 1 else ci_upper,
            standard_error=se * 100 if value > 1 else se,
        )

    def _compare_variants(
        self,
        control: VariantMetricResult,
        treatment: VariantMetricResult,
        metric_name: str,
        confidence_level: float,
    ) -> MetricComparison:
        """Compare treatment to control for a metric."""
        # Calculate lift
        if control.value == 0:
            relative_lift = 0.0 if treatment.value == 0 else float('inf')
        else:
            relative_lift = ((treatment.value - control.value) / control.value) * 100

        absolute_lift = treatment.value - control.value

        # Perform z-test for proportions
        p_value = self._two_proportion_z_test(
            treatment.value / 100 if treatment.value > 1 else treatment.value,
            treatment.sample_size,
            control.value / 100 if control.value > 1 else control.value,
            control.sample_size,
        )

        # Determine significance
        if p_value < 0.01:
            significance = StatisticalSignificance.HIGH
        elif p_value < 0.05:
            significance = StatisticalSignificance.MEDIUM
        elif p_value < 0.1:
            significance = StatisticalSignificance.LOW
        else:
            significance = StatisticalSignificance.NOT_SIGNIFICANT

        # Is treatment better with statistical significance?
        is_winner = (
            significance in (StatisticalSignificance.HIGH, StatisticalSignificance.MEDIUM)
            and treatment.value > control.value
        )

        # Calculate confidence interval for difference
        diff_ci = self._difference_confidence_interval(
            treatment.value / 100 if treatment.value > 1 else treatment.value,
            treatment.sample_size,
            control.value / 100 if control.value > 1 else control.value,
            control.sample_size,
            confidence_level,
        )

        return MetricComparison(
            metric_name=metric_name,
            control_value=control.value,
            treatment_value=treatment.value,
            absolute_lift=absolute_lift,
            relative_lift=relative_lift,
            p_value=p_value,
            significance=significance,
            confidence_interval=diff_ci,
            is_winner=is_winner,
        )

    def _determine_winner(
        self,
        experiment: Experiment,
        primary_metric: MetricType,
        confidence_level: float,
    ) -> Optional[ExperimentVariant]:
        """Determine the winning variant based on primary metric."""
        control = experiment.control_variant
        if not control:
            return None

        control_result = self._calculate_variant_metric(control, primary_metric)

        best_variant = None
        best_lift = 0.0

        for treatment in experiment.treatment_variants:
            treatment_result = self._calculate_variant_metric(treatment, primary_metric)

            # Calculate significance
            p_value = self._two_proportion_z_test(
                treatment_result.value / 100 if treatment_result.value > 1 else treatment_result.value,
                treatment_result.sample_size,
                control_result.value / 100 if control_result.value > 1 else control_result.value,
                control_result.sample_size,
            )

            # Check if statistically significant improvement
            if p_value < (1 - confidence_level):
                lift = treatment_result.value - control_result.value
                if lift > best_lift:
                    best_lift = lift
                    best_variant = treatment

        # If no treatment wins, check if control is best
        if not best_variant:
            for treatment in experiment.treatment_variants:
                treatment_result = self._calculate_variant_metric(treatment, primary_metric)
                if treatment_result.value > control_result.value:
                    return None  # No clear winner yet

            # Control is best
            return control

        return best_variant

    def _generate_recommendation(
        self,
        experiment: Experiment,
        result: ExperimentResult,
    ) -> str:
        """Generate a recommendation based on analysis results."""
        if experiment.total_participants < experiment.min_sample_size:
            return (
                f"Insufficient sample size. Need at least {experiment.min_sample_size} "
                f"participants (currently {experiment.total_participants}). "
                "Continue running the experiment."
            )

        if not result.winning_variant_id:
            # Check if we need more data
            significant_comparisons = [
                c for c in result.metric_comparisons
                if c.significance in (StatisticalSignificance.HIGH, StatisticalSignificance.MEDIUM)
            ]

            if not significant_comparisons:
                return (
                    "No statistically significant differences detected. "
                    "Consider extending the experiment or increasing traffic."
                )
            else:
                return (
                    "Mixed results across metrics. Review individual metric "
                    "comparisons to make a decision."
                )

        # We have a winner
        primary_comparison = next(
            (c for c in result.metric_comparisons if c.is_winner),
            None
        )

        if primary_comparison:
            return (
                f"Recommend implementing '{result.winning_variant_name}'. "
                f"Shows {primary_comparison.relative_lift:.1f}% improvement in "
                f"{primary_comparison.metric_name} (p={primary_comparison.p_value:.4f})."
            )

        return f"'{result.winning_variant_name}' is the best performing variant."

    # Statistical helper methods

    def _calculate_confidence_interval(
        self,
        p: float,
        n: int,
        confidence: float = 0.95,
    ) -> Tuple[float, float, float]:
        """Calculate confidence interval for a proportion."""
        if n == 0:
            return 0.0, 0.0, 0.0

        z = self._z_score(confidence)
        se = math.sqrt(p * (1 - p) / n) if p > 0 and p < 1 else 0

        lower = max(0, p - z * se)
        upper = min(1, p + z * se)

        return lower, upper, se

    def _two_proportion_z_test(
        self,
        p1: float,
        n1: int,
        p2: float,
        n2: int,
    ) -> float:
        """Perform two-proportion z-test."""
        if n1 == 0 or n2 == 0:
            return 1.0

        # Pooled proportion
        p_pooled = (p1 * n1 + p2 * n2) / (n1 + n2)

        if p_pooled == 0 or p_pooled == 1:
            return 1.0

        # Standard error
        se = math.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))

        if se == 0:
            return 1.0

        # Z statistic
        z = (p1 - p2) / se

        # Two-tailed p-value
        p_value = 2 * (1 - self._normal_cdf(abs(z)))

        return p_value

    def _difference_confidence_interval(
        self,
        p1: float,
        n1: int,
        p2: float,
        n2: int,
        confidence: float,
    ) -> Tuple[float, float]:
        """Calculate confidence interval for difference between proportions."""
        if n1 == 0 or n2 == 0:
            return 0.0, 0.0

        diff = p1 - p2
        se = math.sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)
        z = self._z_score(confidence)

        return diff - z * se, diff + z * se

    def _z_score(self, confidence: float) -> float:
        """Get z-score for confidence level."""
        # Common z-scores
        z_scores = {
            0.90: 1.645,
            0.95: 1.96,
            0.99: 2.576,
        }
        return z_scores.get(confidence, 1.96)

    def _normal_cdf(self, z: float) -> float:
        """Approximate the standard normal CDF."""
        # Approximation using error function
        return 0.5 * (1 + math.erf(z / math.sqrt(2)))

    # Sample size calculations

    def calculate_required_sample_size(
        self,
        baseline_rate: float,
        minimum_detectable_effect: float,
        confidence_level: float = 0.95,
        power: float = 0.80,
        num_variants: int = 2,
    ) -> int:
        """
        Calculate required sample size per variant.

        Args:
            baseline_rate: Expected conversion/success rate (e.g., 0.10 for 10%)
            minimum_detectable_effect: Minimum relative effect to detect (e.g., 0.10 for 10%)
            confidence_level: Statistical confidence level
            power: Statistical power (1 - probability of Type II error)
            num_variants: Number of variants including control

        Returns:
            Required sample size per variant
        """
        alpha = 1 - confidence_level
        beta = 1 - power

        # Z-scores
        z_alpha = self._z_score(1 - alpha/2)  # Two-tailed
        z_beta = self._z_score(power)

        # Expected rates
        p1 = baseline_rate
        p2 = baseline_rate * (1 + minimum_detectable_effect)

        # Pooled variance
        p_bar = (p1 + p2) / 2
        variance = 2 * p_bar * (1 - p_bar)

        # Sample size formula
        effect_size = abs(p2 - p1)
        n = ((z_alpha + z_beta) ** 2 * variance) / (effect_size ** 2)

        # Bonferroni correction for multiple comparisons
        n = n * math.log(num_variants)

        return int(math.ceil(n))

    def calculate_experiment_duration(
        self,
        required_sample_size: int,
        daily_traffic: int,
        traffic_percentage: float = 100.0,
        num_variants: int = 2,
    ) -> int:
        """
        Estimate experiment duration in days.

        Args:
            required_sample_size: Required sample per variant
            daily_traffic: Expected daily traffic/calls
            traffic_percentage: Percentage of traffic in experiment
            num_variants: Number of variants

        Returns:
            Estimated days to reach sample size
        """
        effective_daily = daily_traffic * (traffic_percentage / 100)
        per_variant_daily = effective_daily / num_variants

        if per_variant_daily == 0:
            return 0

        return int(math.ceil(required_sample_size / per_variant_daily))
