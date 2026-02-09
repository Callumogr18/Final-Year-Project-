import psycopg2 as pg
import logging

logger = logging.getLogger(__name__)

class ResponseManager:
    def __init__(self, db_session):
        self.db = db_session
        self.cursor = db_session.cursor()

        logger.info("ResponseManager intialised")

    def save_generations(self, response_data):
        logger.info("Saving response and metadata to DB...")

        try:
            self.cursor.execute(
                f'INSERT INTO public.generations ("response_id", "prompt_id", "model_name",' \
                '"llm_response", "latency", "tokens_generated", "tokens_prompt", "total_tokens", "created_at") ' \
                'VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())',
                (
                    response_data["response_id"],
                    response_data["prompt_id"],
                    response_data["model_name"],
                    response_data["llm_response"],
                    response_data["latency_ms"],
                    response_data["tokens_generated"],
                    response_data["tokens_prompt"],
                    response_data["total_tokens"]
                )
            )   

        except Exception as e:
            logger.error(f"Error saving response to DB: {e}")
            exit(1)

        self.db.commit()

