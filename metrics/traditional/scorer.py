from sacrebleu import BLEU
from rouge import Rouge
from uuid import uuid4

bleu = BLEU()
rouge = Rouge()

def metric_scorer(response, reference, conn, prompt_id, response_id, batch_id=None):
    bleu_score = bleu.sentence_score(
        hypothesis=response,
        references=[reference[0]],
    )

    rouge_score = rouge.get_scores(
        hyps=response,
        refs=reference[0]
    )
    r1 = rouge_score[0]['rouge-1']['f']
    r2 = rouge_score[0]['rouge-2']['f']
    rl = rouge_score[0]['rouge-l']['f']

    scores = [bleu_score.score / 100, r1, r2, rl]
    save_scores(scores, response_id, prompt_id, conn, batch_id)

    return bleu_score.score, r1, r2, rl


def save_scores(scores, response_id, prompt_id, conn, batch_id=None):
    cursor = conn.cursor()

    cursor.execute(
        'INSERT INTO metrics ("score_id", "response_id", "prompt_id", "bleu", "rouge_1", "rouge_2", "rouge_l", "batch_id")'
        ' VALUES (%s, %s, %s, %s, %s, %s, %s, %s)',
        (
            str(uuid4()),
            response_id,
            prompt_id,
            scores[0],
            scores[1],
            scores[2],
            scores[3],
            str(batch_id) if batch_id else None
        )
    )
    conn.commit()

