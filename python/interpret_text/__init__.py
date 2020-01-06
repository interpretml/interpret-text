# Setup logging infrustructure
import logging
import os
import atexit
# Only log to disk if environment variable specified
interpret_c_logs = os.environ.get('INTERPRET_TEXT_LOGS')
if interpret_c_logs is not None:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    os.makedirs(os.path.dirname(interpret_c_logs), exist_ok=True)
    handler = logging.FileHandler(interpret_c_logs, mode='w')
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.info('Initializing logging file for interpret-community')

    def close_handler():
        handler.close()
        logger.removeHandler(handler)
    atexit.register(close_handler)
