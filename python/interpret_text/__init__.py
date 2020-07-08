# Setup logging infrustructure
import logging
import os
import atexit

_major = "0"
_minor = "1"
_patch = "2"

__name__ = "interpret-text"
__version__ = "{}.{}.{}.dev3".format(_major, _minor, _patch)

# Only log to disk if environment variable specified
interpret_text_logs = os.environ.get('INTERPRET_TEXT_LOGS')
if interpret_text_logs is not None:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    os.makedirs(os.path.dirname(interpret_text_logs), exist_ok=True)
    handler = logging.FileHandler(interpret_text_logs, mode='w')
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.info('Initializing logging file for interpret-text')

    def close_handler():
        handler.close()
        logger.removeHandler(handler)
    atexit.register(close_handler)
