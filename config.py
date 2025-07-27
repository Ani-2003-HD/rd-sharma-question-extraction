# RD Sharma Question Extraction Pipeline Configuration

# PDF Settings
PDF_PATH = "rd_sharma_class12.pdf"  # Update this path
DEFAULT_CHAPTER = "30"
DEFAULT_TOPIC = "Conditional Probability"

# Model Settings
USE_FREE_MODEL = True
MODEL_NAME = "gpt2"  # Free alternative
DEVICE = "cpu"  # Use CPU for free resources

# Output Settings
OUTPUT_DIR = "sample_output"
DEFAULT_OUTPUT_FILE = "extracted_questions.tex"

# Processing Settings
MAX_QUESTIONS_PER_TOPIC = 50
MIN_CONFIDENCE_SCORE = 0.6
ENABLE_OCR = True

# Logging
LOG_LEVEL = "INFO"
LOG_FILE = "logs/pipeline.log"
