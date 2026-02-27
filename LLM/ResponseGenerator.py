import os
from dotenv import load_dotenv
import logging
from uuid import uuid4

load_dotenv()

logger = logging.getLogger(__name__)

class ResponseGenerator:
    def __init__(self, prompt, clients):
        self.prompt = prompt
        self.generations = [client.generate(prompt) for client in clients]