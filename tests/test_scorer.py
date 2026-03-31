"""
Unit tests for metrics/traditional/scorer.py

DB writes are mocked so no live connection is required.
Short, known strings are used where BLEU/ROUGE values are predictable.
"""
import pytest
from unittest.mock import MagicMock, call

from metrics.traditional.scorer import metric_scorer


def run_scorer(response, reference, task_type='QA', mock_conn=None):
    """Run metric_scorer with a fresh mock conn unless one is provided."""
    if mock_conn is None:
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = MagicMock()
    metric_scorer(
        response=response,
        reference=reference,
        conn=mock_conn,
        prompt_id=1,
        response_id='test-uuid-0001',
        task_type=task_type,
    )
    return mock_conn


class TestScorerOutputValues:

    def test_identical_strings_bleu_near_one(self):
        """Identical hypothesis and reference should yield BLEU close to 1.0."""
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value = cursor

        metric_scorer(
            response='the cat sat on the mat near the door',
            reference='the cat sat on the mat near the door',
            conn=conn,
            prompt_id=1,
            response_id='uuid-1',
            task_type='QA',
        )

        args = cursor.execute.call_args[0][1]
        bleu_score = args[3]
        assert abs(bleu_score - 1.0) < 0.01, f'Expected BLEU ≈ 1.0, got {bleu_score}'

    def test_identical_strings_rouge1_near_one(self):
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value = cursor

        metric_scorer(
            response='the cat sat on the mat near the door',
            reference='the cat sat on the mat near the door',
            conn=conn,
            prompt_id=1,
            response_id='uuid-2',
            task_type='QA',
        )

        args = cursor.execute.call_args[0][1]
        rouge1 = args[4]
        assert abs(rouge1 - 1.0) < 0.01, f'Expected ROUGE-1 ≈ 1.0, got {rouge1}'

    def test_completely_different_strings_rouge1_zero(self):
        """No token overlap should yield ROUGE-1 = 0.0."""
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value = cursor

        metric_scorer(
            response='alpha beta gamma delta',
            reference='the cat sat on the mat',
            conn=conn,
            prompt_id=1,
            response_id='uuid-3',
            task_type='QA',
        )

        args = cursor.execute.call_args[0][1]
        rouge1 = args[4]
        assert rouge1 == 0.0, f'Expected ROUGE-1 = 0.0, got {rouge1}'

    def test_partial_overlap_bleu_between_zero_and_one(self):
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value = cursor

        metric_scorer(
            response='the cat sat on the table',
            reference='the cat sat on the mat near the door',
            conn=conn,
            prompt_id=1,
            response_id='uuid-4',
            task_type='QA',
        )

        args = cursor.execute.call_args[0][1]
        bleu_score = args[3]
        assert 0.0 < bleu_score < 1.0, f'Expected 0 < BLEU < 1, got {bleu_score}'


class TestScorerDBInteraction:

    def test_db_execute_called_once(self):
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value = cursor

        metric_scorer(
            response='Paris',
            reference='Paris',
            conn=conn,
            prompt_id=1,
            response_id='uuid-8',
            task_type='QA',
        )

        cursor.execute.assert_called_once()

    def test_db_commit_called_once(self):
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value = cursor

        metric_scorer(
            response='Paris',
            reference='Paris',
            conn=conn,
            prompt_id=1,
            response_id='uuid-9',
            task_type='QA',
        )

        conn.commit.assert_called_once()

    def test_correct_columns_written(self):
        """Verify that all four metric scores are written in the correct positions."""
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value = cursor

        metric_scorer(
            response='the cat sat on the mat near the door',
            reference='the cat sat on the mat near the door',
            conn=conn,
            prompt_id=1,
            response_id='uuid-10',
            task_type='QA',
        )

        args = cursor.execute.call_args[0][1]
        # (score_id, response_id, prompt_id, bleu, rouge_1, rouge_2, rouge_l, batch_id, task_type)
        assert args[1] == 'uuid-10'   # response_id
        assert args[2] == 1           # prompt_id
        assert args[8] == 'QA'        # task_type
        # All four metric scores are floats in [0, 1] (small epsilon for float precision)
        for i in range(3, 7):
            assert isinstance(args[i], float)
            assert 0.0 <= args[i] <= 1.0 + 1e-9
