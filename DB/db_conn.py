import os
from dotenv import load_dotenv
import psycopg2 as pg
import logging
import pandas as pd

load_dotenv()

logger = logging.getLogger(__name__)

def get_connection():
    try:
        logger.info('Connecting to DB...')
        conn = pg.connect(os.getenv("DB_URL"))
        logger.info('Successfully connected to DB')
        return conn
    except Exception as error:
        logger.error(f'Error during DB connection: {error}')
        return None