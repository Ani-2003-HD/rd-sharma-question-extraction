#!/usr/bin/env python3
"""
RD Sharma Question Extraction - Web Interface Runner
Simple script to start the Flask web application
"""

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import flask
        import fitz
        import transformers
        import torch
        print("✅ All core dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("💡 Please run: pip install -r requirements.txt")
        return False

def check_pdf_exists():
    """Check if the RD Sharma PDF exists"""
    pdf_path = 'data/rd_sharma_class12.pdf'
    if os.path.exists(pdf_path):
        print(f"✅ RD Sharma PDF found: {pdf_path}")
        return True
    else:
        print(f"⚠️  RD Sharma PDF not found: {pdf_path}")
        print("💡 Please ensure the PDF is placed in the data/ directory")
        return False

def create_directories():
    """Create necessary directories"""
    directories = ['output_tex_files', 'logs']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    print("✅ Directories created")

def main():
    """Main function to run the web interface"""
    print("🚀 RD Sharma Question Extraction - Web Interface")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Check if PDF exists
    if not check_pdf_exists():
        print("\n📝 Note: You can still test the interface, but processing will fail without the PDF.")
        print("💡 Download RD Sharma Class 12 PDF and place it in data/rd_sharma_class12.pdf")
    
    # Create directories
    create_directories()
    
    # Check if app.py exists
    if not os.path.exists('app.py'):
        print("❌ app.py not found. Please ensure you're in the correct directory.")
        return
    
    print("\n📱 Starting web interface...")
    print("🌐 The application will be available at: http://localhost:5000")
    print("💡 Select chapter and topic, then watch the magic happen!")
    print("💡 Press Ctrl+C to stop the server")
    print("\n" + "=" * 50)
    
    try:
        # Start Flask app
        from app import app
        print("✅ Flask application loaded successfully")
        
        # Open browser after a short delay
        def open_browser():
            time.sleep(2)
            webbrowser.open('http://localhost:5000')
        
        import threading
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        # Run the app
        app.run(debug=True, host='0.0.0.0', port=5000)
        
    except KeyboardInterrupt:
        print("\n\n👋 Web interface stopped. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error starting web interface: {e}")
        print("💡 Make sure all dependencies are installed: pip install -r requirements.txt")

if __name__ == "__main__":
    main() 