#!/usr/bin/env python3
"""
RD Sharma Question Extraction - Web Interface
A user-friendly web application for the RAG pipeline
"""

from flask import Flask, render_template, request, jsonify, send_file, flash, redirect, url_for
import os
import tempfile
import json
from datetime import datetime
import threading
import time

# Import our pipeline
from rd_sharma_pipeline import RAGPipeline, ExtractedQuestion

app = Flask(__name__)
app.secret_key = 'rd_sharma_extraction_2024'  # For flash messages

# Configuration
OUTPUT_FOLDER = 'output_tex_files'
PDF_PATH = 'data/rd_sharma_class12.pdf'  # Fixed PDF path

# Create directories if they don't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Global variables for job tracking
active_jobs = {}
job_results = {}

@app.route('/')
def index():
    """Main page with chapter and topic selection"""
    return render_template('index.html')

@app.route('/extract', methods=['POST'])
def extract_questions():
    """Handle chapter and topic selection and start processing"""
    chapter = request.form.get('chapter', '').strip()
    topic = request.form.get('topic', '').strip()
    
    if not chapter or not topic:
        flash('Please provide both chapter number and topic', 'error')
        return redirect(url_for('index'))
    
    # Validate chapter number
    try:
        chapter_num = int(chapter)
        if chapter_num < 1 or chapter_num > 100:
            flash('Chapter number must be between 1 and 100', 'error')
            return redirect(url_for('index'))
    except ValueError:
        flash('Please enter a valid chapter number', 'error')
        return redirect(url_for('index'))
    
    # Check if PDF exists
    if not os.path.exists(PDF_PATH):
        flash('RD Sharma PDF not found. Please ensure the PDF is in the data/ directory.', 'error')
        return redirect(url_for('index'))
    
    # Generate job ID
    job_id = f"job_{int(time.time())}"
    
    # Start processing in background
    thread = threading.Thread(target=process_pdf, args=(job_id, chapter, topic))
    thread.daemon = True
    thread.start()
    
    active_jobs[job_id] = {
        'status': 'processing',
        'start_time': datetime.now(),
        'filename': 'rd_sharma_class12.pdf',
        'chapter': chapter,
        'topic': topic
    }
    
    flash(f'Processing started! Job ID: {job_id}', 'success')
    return redirect(url_for('status', job_id=job_id))

def process_pdf(job_id, chapter, topic):
    """Process PDF in background thread"""
    try:
        # Initialize pipeline with fixed PDF
        pipeline = RAGPipeline(PDF_PATH)
        
        # Extract questions
        questions = pipeline.extract_questions(chapter, topic)
        
        # Generate output filename
        clean_topic = topic.replace(' ', '_').lower()
        output_filename = f"chapter_{chapter}_{clean_topic}_questions_{job_id}.tex"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        
        # Save results
        pipeline.save_questions_to_latex(questions, output_path)
        
        # Store results
        job_results[job_id] = {
            'questions': [
                {
                    'text': q.question_text,
                    'latex': q.latex_format,
                    'type': q.question_type,
                    'confidence': q.confidence_score
                } for q in questions
            ],
            'output_file': output_filename,
            'total_questions': len(questions),
            'avg_confidence': sum(q.confidence_score for q in questions) / len(questions) if questions else 0
        }
        
        # Update job status
        active_jobs[job_id]['status'] = 'completed'
        active_jobs[job_id]['end_time'] = datetime.now()
        
    except Exception as e:
        active_jobs[job_id]['status'] = 'failed'
        active_jobs[job_id]['error'] = str(e)
        active_jobs[job_id]['end_time'] = datetime.now()

@app.route('/status/<job_id>')
def status(job_id):
    """Show processing status"""
    if job_id not in active_jobs:
        flash('Job not found', 'error')
        return redirect(url_for('index'))
    
    job = active_jobs[job_id]
    return render_template('status.html', job_id=job_id, job=job)

@app.route('/api/status/<job_id>')
def api_status(job_id):
    """API endpoint for job status"""
    if job_id not in active_jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = active_jobs[job_id]
    response = {
        'status': job['status'],
        'filename': job['filename'],
        'chapter': job['chapter'],
        'topic': job['topic'],
        'start_time': job['start_time'].isoformat() if 'start_time' in job else None
    }
    
    if job['status'] == 'completed':
        results = job_results.get(job_id, {})
        response.update({
            'total_questions': results.get('total_questions', 0),
            'avg_confidence': results.get('avg_confidence', 0),
            'output_file': results.get('output_file', ''),
            'questions': results.get('questions', [])[:5]  # First 5 questions for preview
        })
    elif job['status'] == 'failed':
        response['error'] = job.get('error', 'Unknown error')
    
    return jsonify(response)

@app.route('/results/<job_id>')
def results(job_id):
    """Show detailed results"""
    if job_id not in active_jobs or job_id not in job_results:
        flash('Results not found', 'error')
        return redirect(url_for('index'))
    
    job = active_jobs[job_id]
    results = job_results[job_id]
    
    return render_template('results.html', 
                         job_id=job_id, 
                         job=job, 
                         results=results)

@app.route('/download/<job_id>')
def download_file(job_id):
    """Download the generated LaTeX file"""
    if job_id not in job_results:
        flash('File not found', 'error')
        return redirect(url_for('index'))
    
    output_file = job_results[job_id]['output_file']
    file_path = os.path.join(OUTPUT_FOLDER, output_file)
    
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True, download_name=output_file)
    else:
        flash('File not found', 'error')
        return redirect(url_for('index'))

@app.route('/demo')
def demo():
    """Demo page showing sample results"""
    return render_template('demo.html')

@app.route('/about')
def about():
    """About page with project information"""
    return render_template('about.html')

if __name__ == '__main__':
    print("🚀 Starting RD Sharma Question Extraction Web Interface...")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("💡 Select chapter and topic, then watch the magic happen!")
    app.run(debug=True, host='0.0.0.0', port=5000) 
    app.run(debug=True, host='0.0.0.0', port=5000) 