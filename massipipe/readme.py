# Imports
import logging
from pathlib import Path

# Get logger
logger = logging.getLogger(__name__)

README_TEXT = "Example readme text."


def write_readme(readme_file_path: Path):
    """Write standard readme text to file"""
    try:
        with open(readme_file_path, mode="w", encoding="utf-8") as readme_file:
            readme_file.write(README_TEXT)
    except Exception:
        logger.error(f"Error while writing readme text to {readme_file_path}", exc_info=True)
