import logging
from dotenv import load_dotenv

from metrics.hybrid.scorer import compute_hybrid_score
from DB import db_conn

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    filename="hybrid_eval.log"
)


def extract_prompt(conn, prompt_id):
    cursor = conn.cursor()

    logger.info(f"Extracting prompt {prompt_id} from DB")
    cursor.execute("""
        SELECT jm.hallucination, jm.fluency, jm.consistency,
               jm.reasoning, jm.coherence, jm.accuracy,
               g.model_name, jm.task_type,
               g.llm_response,
               p.ground_truths, p.highlights
        FROM judge_metrics jm
        JOIN generations g ON jm.response_id = g.response_id
        JOIN prompts p ON jm.prompt_id = p.id
        WHERE jm.prompt_id = %s
    """, (prompt_id,))
    rows = cursor.fetchall()

    return rows


if __name__ == '__main__':
    prompt_id = str(input("Enter PromptID >>> "))

    conn = db_conn.get_connection()
    if conn is None:
        logger.error("DB Connection not established")
        exit()

    rows = extract_prompt(conn, prompt_id)

    if not rows:
        print(f"No data found for prompt {prompt_id}")
        exit()

    for row in rows:
        judge = {
            'hallucination': float(row[0]), 'fluency': float(row[1]),
            'consistency': float(row[2]), 'reasoning': float(row[3]),
            'coherence': float(row[4]), 'factual_accuracy': float(row[5])
        }
        model_name = row[6]
        task_type = row[7]
        response_text = row[8]
        ground_truths = row[9]
        highlights = row[10]

        if task_type.upper() == 'SUMMARISATION':
            reference_text = highlights
        else:
            reference_text = ground_truths[0] if isinstance(ground_truths, list) else ground_truths

        result = compute_hybrid_score(response_text, reference_text, judge, task_type)
        print(f"{model_name}: HybridEval={result['hybrid_score']:.4f}  "
              f"Sim={result['similarity']:.4f}  "
              f"Q={result['quality']:.4f}  "
              f"Gate={result['hallucination_gate']:.4f}")
