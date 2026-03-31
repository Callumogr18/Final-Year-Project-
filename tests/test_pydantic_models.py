"""
Unit tests for LLM/judge/pydantic_models.py

Covers MetricEvaluation.score() and yes_count() 
"""
import pytest

from LLM.judge.pydantic_models import SubQuestionAnswer, MetricEvaluation
from tests.conftest import make_metric_evaluation


class TestMetricEvaluationScore:

    def test_score_all_yes(self):
        me = make_metric_evaluation('Hallucination', [True, True, True, True])
        assert me.score() == 1.0

    def test_score_all_no(self):
        me = make_metric_evaluation('Hallucination', [False, False, False, False])
        assert me.score() == 0.0

    def test_score_mixed_3_yes(self):
        me = make_metric_evaluation('Fluency', [True, True, True, False])
        assert me.score() == 0.75

    def test_score_mixed_1_yes(self):
        me = make_metric_evaluation('Reasoning', [True, False, False, False])
        assert me.score() == 0.25

    def test_score_half(self):
        me = make_metric_evaluation('Coherence', [True, True, False, False])
        assert me.score() == 0.5

    def test_score_empty_answers_returns_zero(self):
        """Guard against ZeroDivisionError when answers list is empty."""
        me = MetricEvaluation(metric='Hallucination', answers=[])
        assert me.score() == 0.0

    def test_score_single_yes(self):
        me = make_metric_evaluation('Fluency', [True])
        assert me.score() == 1.0

    def test_score_single_no(self):
        me = make_metric_evaluation('Fluency', [False])
        assert me.score() == 0.0

    def test_score_is_float(self):
        me = make_metric_evaluation('Consistency', [True, False, True, False])
        assert isinstance(me.score(), float)

    def test_score_within_range(self):
        for bools in [
            [True, True, True, True],
            [False, False, False, False],
            [True, False, True, False],
        ]:
            me = make_metric_evaluation('Factual Accuracy', bools)
            assert 0.0 <= me.score() <= 1.0


class TestMetricEvaluationYesCount:

    def test_yes_count_all_yes(self):
        me = make_metric_evaluation('Hallucination', [True, True, True, True])
        assert me.yes_count() == 4

    def test_yes_count_all_no(self):
        me = make_metric_evaluation('Hallucination', [False, False, False, False])
        assert me.yes_count() == 0

    def test_yes_count_two_yes(self):
        me = make_metric_evaluation('Reasoning', [True, False, True, False])
        assert me.yes_count() == 2

    def test_yes_count_matches_score(self):
        """yes_count / len(answers) must equal score()."""
        me = make_metric_evaluation('Coherence', [True, True, False, False])
        assert me.yes_count() / len(me.answers) == me.score()
