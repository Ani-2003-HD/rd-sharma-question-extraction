#!/usr/bin/env python3
"""
Test suite for RD Sharma Question Extraction Pipeline
Basic tests to make sure everything works properly.
"""

import unittest
import tempfile
import os
import sys
from unittest.mock import patch, MagicMock

# Add the current directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rd_sharma_pipeline import (
    ExtractedQuestion, 
    DocumentProcessor, 
    QuestionExtractor, 
    RAGPipeline
)

class TestExtractedQuestion(unittest.TestCase):
    """Test the ExtractedQuestion data class"""
    
    def test_question_creation(self):
        """Test creating an ExtractedQuestion object"""
        question = ExtractedQuestion(
            question_text="Find the probability of getting a head when tossing a coin.",
            latex_format="Find the probability of getting a head when tossing a coin.",
            question_type="practice",
            chapter="30",
            topic="Probability",
            confidence_score=0.85
        )
        
        self.assertEqual(question.question_text, "Find the probability of getting a head when tossing a coin.")
        self.assertEqual(question.question_type, "practice")
        self.assertEqual(question.chapter, "30")
        self.assertEqual(question.topic, "Probability")
        self.assertEqual(question.confidence_score, 0.85)

class TestDocumentProcessor(unittest.TestCase):
    """Test the DocumentProcessor class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.processor = DocumentProcessor()
    
    def test_page_finding_logic(self):
        """Test the logic for finding relevant pages"""
        # Mock document with some pages
        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 10
        
        # Create mock pages with different content
        mock_pages = []
        for i in range(10):
            mock_page = MagicMock()
            if i == 3:
                # Page 3 has chapter 30 content
                mock_page.get_text.return_value = "Chapter 30: Conditional Probability"
            elif i == 7:
                # Page 7 has conditional probability content
                mock_page.get_text.return_value = "conditional probability examples"
            else:
                # Other pages have random content
                mock_page.get_text.return_value = f"Page {i} content"
            mock_pages.append(mock_page)
        
        mock_doc.__getitem__.side_effect = lambda x: mock_pages[x]
        
        # Test finding pages for chapter 30 and conditional probability
        relevant_pages = self.processor._find_relevant_pages(mock_doc, "30", "Conditional Probability")
        
        # Should find pages 3 and 7
        self.assertIn(3, relevant_pages)
        self.assertIn(7, relevant_pages)
        self.assertEqual(len(relevant_pages), 2)

class TestQuestionExtractor(unittest.TestCase):
    """Test the QuestionExtractor class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.extractor = QuestionExtractor()
    
    def test_pattern_extraction(self):
        """Test pattern-based question extraction"""
        sample_text = """
        Chapter 30: Conditional Probability
        
        Question 1: Find the probability that a randomly selected student from a class of 50 students 
        is a girl, given that the student is studying mathematics.
        
        Question 2: If P(A) = 0.3 and P(B|A) = 0.6, find P(A ∩ B).
        
        Some theory content here that should not be extracted.
        
        Question 3: A bag contains 5 red balls and 3 blue balls. If two balls are drawn without replacement, 
        what is the probability that both are red?
        """
        
        questions = self.extractor._extract_by_patterns(sample_text, "30", "Conditional Probability")
        
        # Should extract at least 2 questions
        self.assertGreaterEqual(len(questions), 2)
        
        # Check that extracted items are actually questions
        for question in questions:
            self.assertIsInstance(question, ExtractedQuestion)
            # Check that it's a valid question (contains math or question keywords)
            question_lower = question.question_text.lower()
            self.assertTrue(
                any(keyword in question_lower for keyword in ["find", "calculate", "prove", "show", "p(", "probability"]) or
                "?" in question.question_text,
                f"Question doesn't contain expected keywords: {question.question_text}"
            )
            self.assertGreater(question.confidence_score, 0.5)
    
    def test_latex_conversion(self):
        """Test mathematical symbol conversion to LaTeX"""
        # Test basic mathematical expressions
        test_cases = [
            ("Find P(A|B)", "Find P(A|B)"),  # Should remain the same
            ("Calculate 3/4", "Calculate \\frac{3}{4}"),  # Fraction conversion
            ("Find x^2", "Find x^{2}"),  # Power conversion
            ("Find x_1", "Find x_{1}"),  # Subscript conversion
        ]
        
        for input_text, expected_output in test_cases:
            result = self.extractor._convert_to_latex(input_text)
            self.assertEqual(result, expected_output)
    
    def test_deduplication(self):
        """Test question deduplication"""
        # Create some duplicate questions
        questions = [
            ExtractedQuestion(
                question_text="Find the probability of getting a head.",
                latex_format="Find the probability of getting a head.",
                question_type="practice",
                chapter="30",
                topic="Probability",
                confidence_score=0.9
            ),
            ExtractedQuestion(
                question_text="Find the probability of getting a head.",  # Duplicate
                latex_format="Find the probability of getting a head.",
                question_type="practice",
                chapter="30",
                topic="Probability",
                confidence_score=0.8
            ),
            ExtractedQuestion(
                question_text="Calculate P(A ∩ B).",  # Different question
                latex_format="Calculate P(A ∩ B).",
                question_type="practice",
                chapter="30",
                topic="Probability",
                confidence_score=0.85
            )
        ]
        
        unique_questions = self.extractor._deduplicate_questions(questions)
        
        # Should have 2 unique questions (duplicate removed)
        self.assertEqual(len(unique_questions), 2)
        
        # Should keep the higher confidence version of the duplicate
        self.assertEqual(unique_questions[0].confidence_score, 0.9)

class TestLaTeXGeneration(unittest.TestCase):
    """Test LaTeX document generation"""
    
    def test_latex_document_structure(self):
        """Test that generated LaTeX has proper structure"""
        # Create some sample questions
        questions = [
            ExtractedQuestion(
                question_text="Find the probability of getting a head.",
                latex_format="Find the probability of getting a head.",
                question_type="practice",
                chapter="30",
                topic="Conditional Probability",
                confidence_score=0.9
            ),
            ExtractedQuestion(
                question_text="Calculate P(A ∩ B).",
                latex_format="Calculate P(A ∩ B).",
                question_type="practice",
                chapter="30",
                topic="Conditional Probability",
                confidence_score=0.85
            )
        ]
        
        # Create a temporary pipeline for testing
        with patch('rd_sharma_pipeline.DocumentProcessor'), \
             patch('rd_sharma_pipeline.QuestionExtractor'):
            
            pipeline = RAGPipeline("dummy.pdf")
            latex_doc = pipeline._generate_latex_document(questions)
            
            # Check for essential LaTeX elements
            self.assertIn(r'\documentclass[12pt]{article}', latex_doc)
            self.assertIn(r'\begin{document}', latex_doc)
            self.assertIn(r'\end{document}', latex_doc)
            self.assertIn('Chapter 30: Conditional Probability', latex_doc)
            self.assertIn('Find the probability of getting a head.', latex_doc)
            self.assertIn('Calculate P(A ∩ B).', latex_doc)

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline"""
    
    def test_end_to_end_with_sample_text(self):
        """Test the complete pipeline with sample text"""
        # Create a temporary PDF file for testing
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_file.write(b'%PDF-1.4\n%Sample PDF content\n')
            tmp_file_path = tmp_file.name
        
        try:
            # Mock the PDF processing to return sample text
            sample_text = [
                """
                Chapter 30: Conditional Probability
                
                Question 1: Find the probability that a randomly selected student from a class of 50 students 
                is a girl, given that the student is studying mathematics.
                
                Question 2: If P(A) = 0.3 and P(B|A) = 0.6, find P(A ∩ B).
                
                Question 3: A bag contains 5 red balls and 3 blue balls. If two balls are drawn without replacement, 
                what is the probability that both are red?
                """
            ]
            
            with patch.object(DocumentProcessor, 'extract_text_from_pdf', return_value=sample_text), \
                 patch.object(QuestionExtractor, '_extract_by_patterns') as mock_extract:
                
                # Mock the question extraction to return some questions
                mock_questions = [
                    ExtractedQuestion(
                        question_text="Find the probability that a randomly selected student from a class of 50 students is a girl, given that the student is studying mathematics.",
                        latex_format="Find the probability that a randomly selected student from a class of 50 students is a girl, given that the student is studying mathematics.",
                        question_type="practice",
                        chapter="30",
                        topic="Conditional Probability",
                        confidence_score=0.9
                    )
                ]
                mock_extract.return_value = mock_questions
                
                # Test the pipeline
                pipeline = RAGPipeline(tmp_file_path)
                questions = pipeline.extract_questions("30", "Conditional Probability")
                
                # Should return some questions
                self.assertGreater(len(questions), 0)
                self.assertIsInstance(questions[0], ExtractedQuestion)
                
        finally:
            # Clean up
            os.unlink(tmp_file_path)

def run_tests():
    """Run all tests and print a summary"""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestExtractedQuestion,
        TestDocumentProcessor,
        TestQuestionExtractor,
        TestLaTeXGeneration,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    if result.wasSuccessful():
        print("\nAll tests passed! ✓")
    else:
        print("\nSome tests failed ✗")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)