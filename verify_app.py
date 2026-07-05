import sys
import unittest

class TestSurgiCoderApp(unittest.TestCase):
    
    def test_imports(self):
        """Verifies that all dependencies can be successfully imported."""
        try:
            import streamlit
            import pandas
            import ollama
            import spacy
            import pysbd
            print("SUCCESS: All library imports successful!")
        except ImportError as e:
            self.fail(f"Failed to import a critical dependency: {e}")

    def test_nlp_extraction(self):
        """Verifies that the NLP engine correctly processes text and applies heuristics."""
        from nlp_engine import extract_entities, classify_entity
        
        # Test heuristic classification
        self.assertEqual(classify_entity("appendectomy"), "PROCEDURE")
        self.assertEqual(classify_entity("acute cholecystitis"), "PATHOLOGY")
        self.assertEqual(classify_entity("gallbladder"), "ANATOMY")
        self.assertEqual(classify_entity("resected"), "PROCEDURE")
        self.assertEqual(classify_entity("bleeding"), "PATHOLOGY")
        
        # Test entity extraction
        test_text = "The patient was prepped for an open appendectomy for acute appendicitis."
        entities = extract_entities(test_text)
        
        self.assertGreater(len(entities), 0)
        # Check that we have a procedure and pathology mapped
        labels = [ent["label"] for ent in entities]
        self.assertTrue("PROCEDURE" in labels or "PATHOLOGY" in labels)
        print(f"SUCCESS: NLP extraction test passed! Found {len(entities)} entities.")

    def test_ai_mock_engine(self):
        """Verifies that the AI engine falls back correctly and delivers valid Pydantic models."""
        from ai_engine import get_mock_analysis, ReportAnalysisResult
        
        # Cholecystectomy mock test
        chole_text = "Standard operative details for cholecystectomy."
        chole_res = get_mock_analysis(chole_text)
        
        self.assertIsInstance(chole_res, ReportAnalysisResult)
        self.assertEqual(chole_res.icd10_codes[0].code, "0FT44ZZ")
        
        # Appendectomy mock test
        app_text = "Appendectomy report."
        app_res = get_mock_analysis(app_text)
        self.assertEqual(app_res.icd10_codes[0].code, "0DTJ0ZZ")
        
        # Fallback test
        fallback_res = get_mock_analysis("Random unrelated report details.")
        self.assertEqual(fallback_res.icd10_codes[0].code, "0HBTXZZ")
        print("SUCCESS: AI engine mock data validation test passed!")

if __name__ == "__main__":
    print("Running SurgiCoder AI Test Suite...")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSurgiCoderApp)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    if not result.wasSuccessful():
        sys.exit(1)
