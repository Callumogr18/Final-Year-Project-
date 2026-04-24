import pytest
from unittest.mock import MagicMock

from metrics.traditional.scorer import metric_scorer

MODELS = ['gpt-4.1-mini', 'phi-4', 'grok-4-fast', 'llama']

REFERENCE = 'the cat sat on the mat near the door'

# Simulated per-model outputs of varying quality.
MODEL_RESPONSES = {
    'gpt-4.1-mini': 'the cat sat on the mat near the door',
    'phi-4':        'the cat sat on the mat',
    'grok-4-fast':  'a cat was on the mat near the door',
    'llama':        'the cat sat on the mat near the entrance',
}


def run(response, reference, task_type='QA'):
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value = cursor
    metric_scorer(
        response=response,
        reference=reference,
        conn=conn,
        prompt_id=1,
        response_id='uuid-test',
        task_type=task_type,
    )
    args = cursor.execute.call_args[0][1]
    return {'bleu': args[3], 'rouge_1': args[4], 'rouge_2': args[5], 'rouge_l': args[6]}


@pytest.mark.parametrize('model', MODELS)
def test_scores_in_valid_range(model):
    """
    Every metric must fall within [0, 1] range
    """
    scores = run(MODEL_RESPONSES[model], REFERENCE)
    for name, value in scores.items():
        assert 0.0 <= value <= 1.0 + 1e-9, f'{model} {name} = {value} outside [0, 1]'


@pytest.mark.parametrize('model', MODELS)
def test_qa_and_summarisation_task_types_both_score(model):
    """
    Scorer must handle both task types defined
    QA & Summarisation task types
    """
    qa = run(MODEL_RESPONSES[model], REFERENCE, task_type='QA')
    summ = run(MODEL_RESPONSES[model], REFERENCE, task_type='SUMMARISATION')
    assert qa['rouge_1'] >= 0.0
    assert summ['rouge_1'] >= 0.0


def test_exact_match_scores_near_one():
    """
    Identical response and reference must produce BLEU and ROUGE-1 near 1.0
    """
    scores = run(REFERENCE, REFERENCE)
    assert scores['bleu'] > 0.99
    assert scores['rouge_1'] > 0.99


def test_mean_rouge1_meets_accuracy_threshold():
    """
    Mean ROUGE-1 across all models must meet the >= 0.10 threshold
    """
    r1_scores = [run(MODEL_RESPONSES[m], REFERENCE)['rouge_1'] for m in MODELS]
    mean_r1 = sum(r1_scores) / len(r1_scores)
    assert mean_r1 >= 0.10, f'Mean ROUGE-1 {mean_r1:.3f} below 0.10 threshold'
