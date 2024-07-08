# To install:
# - create virtual environment and activate it
# - git clone https://github.com/mh-skjelvareid/massipipe.git
# - Install:
#   - Plain: pip install .
#   - Editable: pip install -e .
#   - With development dependencies: pip install -e .[dev]

import logging

# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler()],
)
