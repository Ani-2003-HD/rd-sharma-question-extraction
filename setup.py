#!/usr/bin/env python3
"""
Setup script for RD Sharma Question Extraction Pipeline
Handles installation and initial configuration using free resources only
"""

import os
import sys
import subprocess
import urllib.request
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"📦 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during {description}: {e}")
        print(f"Command output: {e.stdout}")
        print(f"Command error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required, found {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def install_dependencies():
    """Install required packages"""
    print("📦 Installing dependencies...")
    
    # Core dependencies
    core_packages = [
        "PyMuPDF==1.23.8",
        "transformers==4.36.0", 
        "torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu",  # CPU version for free usage
        "nltk==3.8.1",
        "numpy==1.24.3",
        "pandas==2.0.3",
        "regex==2023.10.3"
    ]
    
    for package in core_packages:
        if not run_command(f"pip install {package}", f"Installing {package.split('==')[0]}"):
            return False
    
    # Optional packages (continue even if these fail)
    optional_packages = [
        "easyocr==1.7.0",
        "opencv-python==4.8.1.78",
        "pdfplumber==0.9.0"
    ]
    
    print("\n📦 Installing optional packages (OCR support)...")
    for package in optional_packages:
        run_command(f"pip install {package}", f"Installing {package.split('==')[0]} (optional)")
    
    return True

def download_nltk_data():
    """Download required NLTK data"""
    print("📚 Downloading NLTK data...")
    
    nltk_downloads = ['punkt', 'stopwords', 'averaged_perceptron_tagger']
    
    import nltk
    for data_name in nltk_downloads:
        try:
            nltk.download(data_name, quiet=True)
            print(f"✅ Downloaded NLTK {data_name}")
        except Exception as e:
            print(f"⚠️  Warning: Could not download NLTK {data_name}: {e}")

def setup_directories():
    """Create necessary directories"""
    print("📁 Setting up directories...")
    
    directories = [
        "sample_output",
        "tests", 
        "data",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")

def create_config_file():
    """Create a configuration file"""
    config_content = """# RD Sharma Question Extraction Pipeline Configuration

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
"""
    
    with open("config.py", "w") as f:
        f.write(config_content)
    
    print("✅ Created configuration file: config.py")

def run_basic_test():
    """Run a basic test to verify installation"""
    print("🧪 Running basic functionality test...")
    
    test_code = """
import sys
sys.path.append('.')

try:
    from rd_sharma_pipeline import QuestionExtractor, DocumentProcessor
    
    # Test basic functionality
    extractor = QuestionExtractor()
    processor = DocumentProcessor()
    
    # Test LaTeX conversion
    test_text = "Find x if x^2 = 4"
    latex_result = extractor._convert_to_latex(test_text)
    
    print("✅ Basic functionality test passed")
    print(f"Sample conversion: '{test_text}' -> '{latex_result}'")
    
except Exception as e:
    print(f"❌ Basic test failed: {e}")
    sys.exit(1)
"""
    
    with open("test_installation.py", "w") as f:
        f.write(test_code)
    
    return run_command("python test_installation.py", "Running basic test")

def download_sample_pdf():
    """Provide instructions for downloading the RD Sharma PDF"""
    print("\n📖 PDF Download Instructions:")
    print("=" * 50)
    print("1. Download RD Sharma Class 12 PDF from:")
    print("   https://drive.google.com/file/d/1BQllRXh5_ID08uPTVfEe0DgmxPUm867F/view")
    print("2. Save it as 'rd_sharma_class12.pdf' in this directory")
    print("3. Update the PDF_PATH in config.py if you use a different name/location")
    print("=" * 50)

def main():
    """Main setup function"""
    print("🚀 RD Sharma Question Extraction Pipeline Setup")
    print("=" * 60)
    print("This setup uses FREE resources only - no paid APIs required!")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Dependency installation failed. Please check your internet connection and try again.")
        sys.exit(1)
    
    # Download NLTK data
    download_nltk_data()
    
    # Setup directories
    setup_directories()
    
    # Create config file
    create_config_file()
    
    # Run basic test
    if not run_basic_test():
        print("❌ Installation test failed. Please check the error messages above.")
        sys.exit(1)
    
    # Clean up test file
    if os.path.exists("test_installation.py"):
        os.remove("test_installation.py")
    
    # PDF download instructions
    download_sample_pdf()
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next Steps:")
    print("1. Download the RD Sharma PDF (see instructions above)")
    print("2. Run the demo: python demo_notebook.py")
    print("3. Or use CLI: python rd_sharma_pipeline.py --help")
    print("4. Run tests: python test_pipeline.py")
    
    print("\n💡 Quick Start:")
    print("python rd_sharma_pipeline.py --pdf rd_sharma_class12.pdf --chapter 30 --topic 'Conditional Probability'")

if __name__ == "__main__":
    main()