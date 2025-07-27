# Output LaTeX Files Directory

This directory contains all the generated LaTeX files from the RD Sharma Question Extraction Pipeline.

## Directory Structure

All extracted question files are automatically saved to this directory with the following naming conventions:

### Auto-generated filenames (default):
- `chapter_{chapter_number}_{topic_name}_questions.tex`
- Example: `chapter_30_conditional_probability_questions.tex`

### Custom filenames:
- When using `--output filename.tex`, files are saved as `filename.tex` in this directory

## File Contents

Each `.tex` file contains:
- **Professional LaTeX formatting** with proper mathematical notation
- **High-confidence questions** (confidence score ≥ 0.90)
- **Color-coded elements** for better readability
- **Mathematical symbols** properly formatted in LaTeX
- **OCR error correction** for clean text output

## Usage

1. **Compile LaTeX files** using any LaTeX editor (TeXstudio, Overleaf, etc.)
2. **View formatted output** as PDF with professional mathematical notation
3. **Use for educational purposes** - all questions are extracted from RD Sharma textbooks

## File Types

- **Integration questions**: Calculus integration problems
- **Probability questions**: Statistical and probability problems  
- **Linear Programming**: Optimization problems
- **Vectors**: Vector algebra and geometry
- **Differential Equations**: ODE and PDE problems

## Quality Assurance

- All files contain high-confidence extractions (≥0.90)
- OCR errors have been corrected
- Mathematical notation is properly formatted
- Professional typography and layout

## Recent Files

- `chapter_30_conditional_probability_questions.tex` - Latest extraction
- `custom_test_output.tex` - Custom filename example
- Various topic-specific extractions from different chapters 