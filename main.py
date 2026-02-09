import logging
from dotenv import load_dotenv

from DB.prompts.PromptManager import PromptManager, PromptBatch, PromptBatcher
from DB.LLM_storage.ResponseManager import ResponseManager
from DB import db_conn
from LLM.ResponseGenerator import ResponseGenerator
from metrics.traditional.scorer import metric_scorer, save_scores

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    filename="app.log"
)

if __name__ == '__main__':
    logger.info("Establishing connection to DB")

    conn = db_conn.get_connection()
    if conn is None:
        logger.error("Failed to connect to DB")
        exit(1)
    
    logger.info("Connection to DB established, loading prompts...")

    prompt_manager = PromptManager(conn)
    llm_manager = ResponseManager(conn)

    task_type = input("Enter task type >>> ")
    prompts = prompt_manager.load_prompts_by_task(task_type)

    if not prompts:
        logger.error(f"No prompts found for task type: {task_type}")
        conn.close()
        exit(1)

    logger.info("Checking if batch possible...")    
    
    # HARDCODED VAL - Address
    if len(prompts) < 10:
        logger.info("Not batching, not enough prompts")

        logger.info("Generating responses from LLMs - No Batching")

        for prompt in prompts:
            generator = ResponseGenerator(prompt, prompt.id)
            llm_manager.save_generations(generator.generation_data)
    
    else:
        batches = prompt_manager.batch_prompts(prompts, batch_size=10)

        logger.info(f"Created {len(batches)} from {len(prompts)}")

        for batch in batches:
            logger.info(f"Processing batch - ID: {batch.batch_id} with {batch.size()}")

            for prompt in batch.prompts:
                generator = ResponseGenerator(prompt, prompt.id)
                llm_manager.save_generations(generator.generation_data)
                bleu, r1, r2, rl = metric_scorer(generator.generation_data['llm_response'],  
                                                prompt.reference_output, conn, prompt.id,
                                                generator.generation_data['response_id'])
        conn.close()
    
