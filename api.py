import os
import logging
from typing import Optional
from uuid import uuid4
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from DB.db_conn import get_connection
from DB.prompts.PromptManager import PromptManager
from DB.LLM_storage.ResponseManager import ResponseManager
from LLM.ResponseGenerator import ResponseGenerator
from LLM.clients import openai_client
from LLM.judge.judge import LLMAsJudge
from LLM.judge.helper import save_judge_scores, scores_to_dict
from metrics.traditional.scorer import metric_scorer
from metrics.hybrid.scorer import compute_hybrid_score

load_dotenv()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

app = FastAPI(title="LLM Evaluation API", version="1.0.0")

# In-memory job store (sufficient for demo/FYP context)
jobs = {}


class EvaluateRequest(BaseModel):
    task_type: Optional[str] = None
    limit: int = 1


class JobStatus(BaseModel):
    job_id: str
    status: str
    message: str
    result_count: Optional[int] = None


@app.get("/health")
def health_check():
    """Liveness check for the service."""
    return {"status": "ok"}


@app.get("/prompts")
def list_prompts(task_type: Optional[str] = None):
    """List available prompts, optionally filtered by task type."""
    conn = get_connection()
    if conn is None:
        raise HTTPException(status_code=503, detail="Database connection failed")

    pm = PromptManager(conn)
    try:
        if task_type:
            prompts = pm.load_prompts_by_task(task_type)
        else:
            prompts = pm.load_prompts_by_task("QA") + pm.load_prompts_by_task("SUMMARISATION")

        return {
            "count": len(prompts),
            "prompts": [
                {
                    "id": p.id,
                    "task_type": p.task_type,
                    "input_text": p.input_text[:100],
                }
                for p in prompts
            ],
        }
    finally:
        conn.close()


@app.post("/evaluate")
def start_evaluation(request: EvaluateRequest, background_tasks: BackgroundTasks):
    """Trigger an evaluation job. Returns immediately with a job ID."""
    job_id = str(uuid4())
    jobs[job_id] = {"status": "running", "message": "Evaluation in progress", "count": 0}

    background_tasks.add_task(
        _evaluate_task,
        job_id=job_id,
        task_type=request.task_type,
        limit=request.limit,
    )

    return {"job_id": job_id, "status": "queued"}


@app.get("/jobs/{job_id}")
def get_job_status(job_id: str):
    """Poll for job completion and results."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    return jobs[job_id]


def _evaluate_task(job_id: str, task_type: Optional[str], limit: int):
    """Background task that runs the evaluation pipeline."""
    conn = get_connection()
    if conn is None:
        jobs[job_id] = {
            "status": "error",
            "message": "Database connection failed",
            "count": 0,
        }
        return

    try:
        pm = PromptManager(conn)
        rm = ResponseManager(conn)
        judge = LLMAsJudge()

        # Load prompts
        if task_type:
            all_prompts = pm.load_prompts_by_task(task_type)
        else:
            all_prompts = pm.load_prompts_by_task("QA") + pm.load_prompts_by_task("SUMMARISATION")

        prompts = all_prompts[:limit]

        # Initialize clients
        clients = [
            openai_client.OpenAIClient(
                os.getenv("GPT_ENDPOINT"),
                os.getenv("API_KEY"),
                os.getenv("GPT_MODEL"),
            ),
            openai_client.OpenAIClient(
                os.getenv("GROK_ENDPOINT"),
                os.getenv("API_KEY"),
                os.getenv("GROK_MODEL"),
            ),
            openai_client.OpenAIClient(
                os.getenv("PHI_ENDPOINT"),
                os.getenv("API_KEY"),
                os.getenv("PHI_MODEL"),
            ),
            openai_client.OpenAIClient(
                os.getenv("LLAMA_ENDPOINT"),
                os.getenv("API_KEY"),
                os.getenv("LLAMA_MODEL"),
            ),
        ]

        result_count = 0

        for prompt in prompts:
            reference = (
                prompt.highlights
                if prompt.task_type.upper() == "SUMMARISATION"
                else prompt.reference_output
            )

            generator = ResponseGenerator(prompt, clients)

            for gen_data in generator.generations:
                if gen_data["llm_response"] is None:
                    logger.warning(
                        f"Skipping prompt {prompt.id} - no response (content filter/API error)"
                    )
                    continue

                # Save response
                rm.save_generations(gen_data)

                # Score with traditional metrics
                metric_scorer(
                    gen_data["llm_response"],
                    reference,
                    conn,
                    prompt.id,
                    gen_data["response_id"],
                    prompt.task_type,
                )

                # Judge evaluation
                judge_result = judge.evaluate(prompt, gen_data["llm_response"])
                save_judge_scores(
                    judge_result,
                    gen_data["response_id"],
                    prompt.id,
                    conn,
                    prompt.task_type,
                )

                # Hybrid score
                judge_dict = scores_to_dict(judge_result)
                ref_text = (
                    reference[0] if isinstance(reference, list) else reference
                )
                hybrid = compute_hybrid_score(
                    gen_data["llm_response"], ref_text, judge_dict, prompt.task_type
                )

                result_count += 1
                jobs[job_id]["count"] = result_count

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["message"] = f"Evaluation complete: {result_count} results"

    except Exception as e:
        logger.error(f"Evaluation job {job_id} failed: {e}")
        jobs[job_id] = {
            "status": "error",
            "message": f"Error: {str(e)}",
            "count": 0,
        }
    finally:
        conn.close()
