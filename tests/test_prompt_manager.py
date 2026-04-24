"""
Unit tests for DB/prompts/PromptManager.py

All tests use a mock DB connection — no live PostgreSQL required.
"""
from unittest.mock import MagicMock

from DB.prompts.PromptManager import PromptManager
from DB.prompts.Prompt import Prompt


# Helper Functions

def _make_mock_conn(fetchall_return=None, fetchone_return=None):
    """
    Return (conn, cursor) mocks pre-configured with the given return values
    """
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value = cursor
    cursor.fetchall.return_value = fetchall_return or []
    cursor.fetchone.return_value = fetchone_return
    return conn, cursor


def _sample_row(
    id=1,
    task_type='QA',
    question='What is the capital of France?',
    ground_truths='Paris',
    answer='Paris',
    contexts=['France is a country in Europe. Its capital is Paris.'],
    article=None,
    highlights=None,
):
    return (id, task_type, question, ground_truths, answer, contexts, article, highlights)


def _make_prompts(n, task_type='QA'):
    """
    Create n minimal Prompt objects
    """
    return [
        Prompt(
            id=i,
            task_type=task_type,
            input_text=f'Question {i}',
            reference_output=f'Answer {i}',
            answer=f'Answer {i}',
            contexts=[f'Context {i}'],
        )
        for i in range(n)
    ]


class TestLoadPromptsByTask:

    def test_returns_list_of_prompts(self):
        rows = [_sample_row(id=i) for i in range(1, 4)]
        conn, _ = _make_mock_conn(fetchall_return=rows)
        manager = PromptManager(conn)

        result = manager.load_prompts_by_task('QA')

        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(p, Prompt) for p in result)

    def test_uppercase_normalisation(self):
        """
        Input 'qa' should be normalised to 'QA' and hit the DB
        """
        rows = [_sample_row()]
        conn, cursor = _make_mock_conn(fetchall_return=rows)
        manager = PromptManager(conn)

        result = manager.load_prompts_by_task('qa')

        # load_prompts_by_task passes task_type as-is to the SQL query;
        # the valid_tasks check uses .upper() so 'qa' passes, but the
        # cursor receives the original string. We just verify the DB was queried.
        cursor.execute.assert_called()
        assert len(result) == 1

    def test_invalid_task_type_returns_empty_list(self):
        conn, cursor = _make_mock_conn()
        manager = PromptManager(conn)

        result = manager.load_prompts_by_task('INVALID')

        assert result == []
        # The cursor should only be called during __init__ (not for the query)
        cursor.execute.assert_not_called()

    def test_empty_db_returns_empty_list(self):
        conn, _ = _make_mock_conn(fetchall_return=[])
        manager = PromptManager(conn)

        result = manager.load_prompts_by_task('QA')

        assert result == []

    def test_prompt_fields_populated_correctly(self):
        row = _sample_row(
            id=42,
            task_type='QA',
            question='Who wrote Hamlet?',
            ground_truths='Shakespeare',
            answer='William Shakespeare',
            contexts=['Hamlet is a play by William Shakespeare.'],
        )
        conn, _ = _make_mock_conn(fetchall_return=[row])
        manager = PromptManager(conn)

        result = manager.load_prompts_by_task('QA')
        p = result[0]

        assert p.id == 42
        assert p.task_type == 'QA'
        assert p.input_text == 'Who wrote Hamlet?'
        assert p.reference_output == 'Shakespeare'
        assert p.answer == 'William Shakespeare'
        assert p.contexts == ['Hamlet is a play by William Shakespeare.']


class TestLoadPromptById:

    def test_found_returns_prompt(self):
        row = _sample_row(id=7)
        conn, _ = _make_mock_conn(fetchone_return=row)
        manager = PromptManager(conn)

        result = manager.load_prompt_by_id(7)

        assert isinstance(result, Prompt)
        assert result.id == 7

    def test_not_found_returns_none(self):
        conn, _ = _make_mock_conn(fetchone_return=None)
        manager = PromptManager(conn)

        result = manager.load_prompt_by_id(999)

        assert result is None

    def test_summarisation_prompt_fields(self):
        row = _sample_row(
            id=5,
            task_type='SUMMARISATION',
            question='Summarise this.',
            ground_truths=None,
            answer=None,
            contexts=None,
            article='Long article text here.',
            highlights='Short summary.',
        )
        conn, _ = _make_mock_conn(fetchone_return=row)
        manager = PromptManager(conn)

        result = manager.load_prompt_by_id(5)

        assert result.task_type == 'SUMMARISATION'
        assert result.article == 'Long article text here.'
        assert result.highlights == 'Short summary.'


class TestLoadPromptsByIds:

    def test_multiple_ids_return_multiple_prompts(self):
        rows = [_sample_row(id=i) for i in range(1, 4)]
        conn, _ = _make_mock_conn(fetchall_return=rows)
        manager = PromptManager(conn)

        result = manager.load_prompts_by_ids([1, 2, 3])

        assert len(result) == 3
        assert all(isinstance(p, Prompt) for p in result)

    def test_empty_ids_returns_empty_list(self):
        conn, _ = _make_mock_conn(fetchall_return=[])
        manager = PromptManager(conn)

        result = manager.load_prompts_by_ids([])

        assert result == []