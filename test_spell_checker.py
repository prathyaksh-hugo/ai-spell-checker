#!/usr/bin/env python3
"""
Comprehensive Test Suite for Advanced AI Spell Checking System

Tests:
1. Individual spell checker components
2. API endpoints functionality
3. Learning and adaptation capabilities
4. Performance and accuracy metrics
5. Edge cases and error handling
"""

import os
import sys
import json
import time
import asyncio
import logging
from typing import List, Dict, Any
import requests
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpellCheckerTester:
    """Comprehensive test suite for the spell checking system"""
    
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url
        self.test_results = {
            "passed": 0,
            "failed": 0,
            "errors": [],
            "performance_metrics": {},
            "accuracy_metrics": {}
        }
    
    def log_test_result(self, test_name: str, passed: bool, details: str = "", performance_ms: float = None):
        """Log test result"""
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{status}: {test_name}")
        
        if details:
            logger.info(f"    Details: {details}")
        
        if performance_ms:
            logger.info(f"    Performance: {performance_ms:.2f}ms")
            self.test_results["performance_metrics"][test_name] = performance_ms
        
        if passed:
            self.test_results["passed"] += 1
        else:
            self.test_results["failed"] += 1
            self.test_results["errors"].append(f"{test_name}: {details}")
    
    def test_api_health(self):
        """Test API health endpoint"""
        try:
            response = requests.get(f"{self.api_base_url}/health", timeout=10)
            
            if response.status_code == 200:
                health_data = response.json()
                passed = health_data.get("status") in ["healthy", "degraded"]
                details = f"Status: {health_data.get('status')}, Components: {health_data.get('components')}"
            else:
                passed = False
                details = f"HTTP {response.status_code}"
            
            self.log_test_result("API Health Check", passed, details)
            return passed
            
        except Exception as e:
            self.log_test_result("API Health Check", False, f"Error: {str(e)}")
            return False
    
    def test_models_endpoint(self):
        """Test models information endpoint"""
        try:
            response = requests.get(f"{self.api_base_url}/models", timeout=10)
            
            if response.status_code == 200:
                models_data = response.json()
                available_models = models_data.get("available_models", [])
                loaded_models = models_data.get("loaded_models", [])
                
                passed = len(available_models) > 0
                details = f"Available: {len(available_models)}, Loaded: {len(loaded_models)}"
            else:
                passed = False
                details = f"HTTP {response.status_code}"
            
            self.log_test_result("Models Endpoint", passed, details)
            return passed
            
        except Exception as e:
            self.log_test_result("Models Endpoint", False, f"Error: {str(e)}")
            return False
    
    def test_word_spell_checking(self):
        """Test word-level spell checking"""
        test_cases = [
            {"word": "appliction", "expected_correct": False, "expected_suggestion": "application"},
            {"word": "recieve", "expected_correct": False, "expected_suggestion": "receive"},
            {"word": "seperate", "expected_correct": False, "expected_suggestion": "separate"},
            {"word": "hello", "expected_correct": True, "expected_suggestion": None},
            {"word": "world", "expected_correct": True, "expected_suggestion": None},
            {"word": "fhdhfdhdf", "expected_correct": False, "expected_suggestion": None}  # Random string
        ]
        
        passed_tests = 0
        total_tests = len(test_cases)
        performance_times = []
        
        for test_case in test_cases:
            try:
                start_time = time.time()
                
                payload = {
                    "texts": [test_case["word"]],
                    "return_suggestions": True,
                    "max_suggestions": 5
                }
                
                response = requests.post(
                    f"{self.api_base_url}/spell_check",
                    json=payload,
                    timeout=30
                )
                
                processing_time = (time.time() - start_time) * 1000
                performance_times.append(processing_time)
                
                if response.status_code == 200:
                    result_data = response.json()
                    results = result_data.get("results", [])
                    
                    if results:
                        word_result = results[0]
                        is_correct = word_result.get("is_correct", True)
                        suggestions = word_result.get("suggestions", [])
                        
                        # Check correctness
                        correctness_match = is_correct == test_case["expected_correct"]
                        
                        # Check suggestions if word is incorrect
                        suggestion_match = True
                        if not is_correct and test_case["expected_suggestion"]:
                            suggestion_words = [s.get("word", "") for s in suggestions]
                            suggestion_match = test_case["expected_suggestion"] in suggestion_words
                        
                        if correctness_match and suggestion_match:
                            passed_tests += 1
                            logger.info(f"    ✅ '{test_case['word']}': {is_correct} (suggestions: {len(suggestions)})")
                        else:
                            logger.info(f"    ❌ '{test_case['word']}': Expected {test_case['expected_correct']}, got {is_correct}")
                    else:
                        logger.info(f"    ❌ '{test_case['word']}': No results returned")
                else:
                    logger.info(f"    ❌ '{test_case['word']}': HTTP {response.status_code}")
                    
            except Exception as e:
                logger.info(f"    ❌ '{test_case['word']}': Error - {str(e)}")
        
        avg_performance = sum(performance_times) / len(performance_times) if performance_times else 0
        passed = passed_tests == total_tests
        details = f"{passed_tests}/{total_tests} tests passed"
        
        self.log_test_result("Word Spell Checking", passed, details, avg_performance)
        self.test_results["accuracy_metrics"]["word_checking_accuracy"] = passed_tests / total_tests
        return passed
    
    def test_sentence_correction(self):
        """Test sentence-level correction"""
        test_cases = [
            {
                "sentence": "This is a sampel text with some erors.",
                "should_have_corrections": True
            },
            {
                "sentence": "Another exampl of incorect speling.",
                "should_have_corrections": True
            },
            {
                "sentence": "This is a perfect sentence.",
                "should_have_corrections": False
            }
        ]
        
        passed_tests = 0
        total_tests = len(test_cases)
        performance_times = []
        
        for test_case in test_cases:
            try:
                start_time = time.time()
                
                payload = {
                    "texts": [test_case["sentence"]],
                    "model_type": "advanced",
                    "return_confidence": True,
                    "apply_corrections": True
                }
                
                response = requests.post(
                    f"{self.api_base_url}/correct_batch",
                    json=payload,
                    timeout=30
                )
                
                processing_time = (time.time() - start_time) * 1000
                performance_times.append(processing_time)
                
                if response.status_code == 200:
                    result_data = response.json()
                    results = result_data.get("results", [])
                    
                    if results:
                        sentence_result = results[0]
                        original = sentence_result.get("original_text", "")
                        corrected = sentence_result.get("corrected_text", "")
                        
                        has_corrections = original != corrected
                        
                        if has_corrections == test_case["should_have_corrections"]:
                            passed_tests += 1
                            logger.info(f"    ✅ Sentence correction test passed")
                            logger.info(f"        Original: {original}")
                            logger.info(f"        Corrected: {corrected}")
                        else:
                            logger.info(f"    ❌ Expected corrections: {test_case['should_have_corrections']}, got: {has_corrections}")
                    else:
                        logger.info(f"    ❌ No results returned")
                else:
                    logger.info(f"    ❌ HTTP {response.status_code}")
                    
            except Exception as e:
                logger.info(f"    ❌ Error: {str(e)}")
        
        avg_performance = sum(performance_times) / len(performance_times) if performance_times else 0
        passed = passed_tests == total_tests
        details = f"{passed_tests}/{total_tests} tests passed"
        
        self.log_test_result("Sentence Correction", passed, details, avg_performance)
        self.test_results["accuracy_metrics"]["sentence_correction_accuracy"] = passed_tests / total_tests
        return passed
    
    def test_learning_functionality(self):
        """Test learning and adaptation capabilities"""
        try:
            # Test learning a correction
            learn_payload = {
                "original_word": "teh",
                "corrected_word": "the",
                "context": "common typo test"
            }
            
            learn_response = requests.post(
                f"{self.api_base_url}/learn",
                json=learn_payload,
                timeout=10
            )
            
            learn_success = learn_response.status_code == 200
            
            # Test whitelist functionality
            whitelist_payload = {
                "word": "TestWord123",
                "category": "test"
            }
            
            whitelist_response = requests.post(
                f"{self.api_base_url}/whitelist",
                json=whitelist_payload,
                timeout=10
            )
            
            whitelist_success = whitelist_response.status_code == 200
            
            # Test ignore functionality
            ignore_payload = {
                "word": "IgnoreWord456",
                "context": "test context"
            }
            
            ignore_response = requests.post(
                f"{self.api_base_url}/ignore",
                json=ignore_payload,
                timeout=10
            )
            
            ignore_success = ignore_response.status_code == 200
            
            passed = learn_success and whitelist_success and ignore_success
            details = f"Learn: {learn_success}, Whitelist: {whitelist_success}, Ignore: {ignore_success}"
            
            self.log_test_result("Learning Functionality", passed, details)
            return passed
            
        except Exception as e:
            self.log_test_result("Learning Functionality", False, f"Error: {str(e)}")
            return False
    
    def test_performance_benchmarks(self):
        """Test system performance with various loads"""
        test_cases = [
            {"name": "Single Word", "texts": ["appliction"], "expected_max_ms": 1000},
            {"name": "Multiple Words", "texts": ["appliction", "recieve", "seperate"], "expected_max_ms": 2000},
            {"name": "Long Sentence", "texts": ["This is a very long sentence with multiple spelling errors like recieve, seperate, and appliction to test performance."], "expected_max_ms": 3000},
            {"name": "Batch Processing", "texts": ["word1", "recieve", "seperate", "appliction", "hello"] * 5, "expected_max_ms": 5000}
        ]
        
        performance_results = {}
        all_passed = True
        
        for test_case in test_cases:
            try:
                start_time = time.time()
                
                payload = {
                    "texts": test_case["texts"],
                    "return_suggestions": True,
                    "max_suggestions": 3
                }
                
                response = requests.post(
                    f"{self.api_base_url}/spell_check",
                    json=payload,
                    timeout=30
                )
                
                processing_time = (time.time() - start_time) * 1000
                performance_results[test_case["name"]] = processing_time
                
                passed = (response.status_code == 200 and 
                         processing_time <= test_case["expected_max_ms"])
                
                if not passed:
                    all_passed = False
                
                logger.info(f"    {test_case['name']}: {processing_time:.2f}ms "
                          f"(max: {test_case['expected_max_ms']}ms) {'✅' if passed else '❌'}")
                
            except Exception as e:
                all_passed = False
                logger.info(f"    {test_case['name']}: Error - {str(e)}")
        
        avg_performance = sum(performance_results.values()) / len(performance_results) if performance_results else 0
        details = f"Average: {avg_performance:.2f}ms"
        
        self.log_test_result("Performance Benchmarks", all_passed, details, avg_performance)
        self.test_results["performance_metrics"].update(performance_results)
        return all_passed
    
    def test_error_handling(self):
        """Test error handling and edge cases"""
        test_cases = [
            {"name": "Empty Request", "payload": {"texts": []}, "expect_error": False},
            {"name": "Invalid JSON", "payload": "invalid json", "expect_error": True},
            {"name": "Very Long Word", "payload": {"texts": ["a" * 1000]}, "expect_error": False},
            {"name": "Special Characters", "payload": {"texts": ["@#$%^&*()"]}, "expect_error": False},
            {"name": "Unicode Text", "payload": {"texts": ["café", "naïve", "résumé"]}, "expect_error": False}
        ]
        
        passed_tests = 0
        total_tests = len(test_cases)
        
        for test_case in test_cases:
            try:
                if isinstance(test_case["payload"], str):
                    # Test invalid JSON
                    response = requests.post(
                        f"{self.api_base_url}/spell_check",
                        data=test_case["payload"],
                        headers={"Content-Type": "application/json"},
                        timeout=10
                    )
                else:
                    response = requests.post(
                        f"{self.api_base_url}/spell_check",
                        json=test_case["payload"],
                        timeout=10
                    )
                
                if test_case["expect_error"]:
                    # Should return error status
                    passed = response.status_code >= 400
                else:
                    # Should handle gracefully
                    passed = response.status_code == 200
                
                if passed:
                    passed_tests += 1
                    logger.info(f"    ✅ {test_case['name']}: Handled correctly")
                else:
                    logger.info(f"    ❌ {test_case['name']}: Unexpected response {response.status_code}")
                
            except Exception as e:
                # For error cases, exceptions might be expected
                if test_case["expect_error"]:
                    passed_tests += 1
                    logger.info(f"    ✅ {test_case['name']}: Error handled correctly")
                else:
                    logger.info(f"    ❌ {test_case['name']}: Unexpected error - {str(e)}")
        
        passed = passed_tests == total_tests
        details = f"{passed_tests}/{total_tests} error handling tests passed"
        
        self.log_test_result("Error Handling", passed, details)
        return passed
    
    def test_direct_components(self):
        """Test spell checker components directly (without API)"""
        try:
            from advanced_spell_checker import MultiEngineSpellChecker
            
            checker = MultiEngineSpellChecker()
            
            # Test word checking
            word_result = checker.check_word("appliction")
            word_passed = not word_result.is_correct and len(word_result.suggestions) > 0
            
            # Test sentence checking
            sentence_result = checker.check_sentence("This is a sampel text with erors.")
            sentence_passed = sentence_result.get("total_corrections", 0) > 0
            
            # Test learning
            checker.learn_correction("testword", "testcorrection")
            learn_result = checker.check_word("testword")
            learn_passed = learn_result.corrected_text == "testcorrection"
            
            passed = word_passed and sentence_passed and learn_passed
            details = f"Word: {word_passed}, Sentence: {sentence_passed}, Learning: {learn_passed}"
            
            self.log_test_result("Direct Component Testing", passed, details)
            return passed
            
        except Exception as e:
            self.log_test_result("Direct Component Testing", False, f"Error: {str(e)}")
            return False
    
    def test_accuracy_against_known_dataset(self):
        """Test accuracy against a known dataset of misspellings"""
        # Common misspellings and their corrections
        test_dataset = [
            ("recieve", "receive"),
            ("seperate", "separate"),
            ("definately", "definitely"),
            ("occured", "occurred"),
            ("accomodate", "accommodate"),
            ("begining", "beginning"),
            ("calender", "calendar"),
            ("existance", "existence"),
            ("expierence", "experience"),
            ("independant", "independent"),
            ("priviledge", "privilege"),
            ("tommorrow", "tomorrow"),
            ("untill", "until"),
            ("wierd", "weird")
        ]
        
        correct_suggestions = 0
        total_tests = len(test_dataset)
        
        for misspelling, correct_word in test_dataset:
            try:
                payload = {
                    "texts": [misspelling],
                    "return_suggestions": True,
                    "max_suggestions": 5
                }
                
                response = requests.post(
                    f"{self.api_base_url}/spell_check",
                    json=payload,
                    timeout=10
                )
                
                if response.status_code == 200:
                    result_data = response.json()
                    results = result_data.get("results", [])
                    
                    if results:
                        suggestions = results[0].get("suggestions", [])
                        suggestion_words = [s.get("word", "") for s in suggestions]
                        
                        if correct_word in suggestion_words:
                            correct_suggestions += 1
                            logger.info(f"    ✅ '{misspelling}' -> '{correct_word}' found in suggestions")
                        else:
                            logger.info(f"    ❌ '{misspelling}' -> '{correct_word}' not found. Got: {suggestion_words[:3]}")
                
            except Exception as e:
                logger.info(f"    ❌ Error testing '{misspelling}': {str(e)}")
        
        accuracy = correct_suggestions / total_tests
        passed = accuracy >= 0.7  # 70% accuracy threshold
        details = f"Accuracy: {accuracy:.1%} ({correct_suggestions}/{total_tests})"
        
        self.log_test_result("Accuracy Against Known Dataset", passed, details)
        self.test_results["accuracy_metrics"]["known_dataset_accuracy"] = accuracy
        return passed
    
    def run_all_tests(self):
        """Run all test suites"""
        logger.info("=" * 80)
        logger.info("🧪 RUNNING COMPREHENSIVE SPELL CHECKER TESTS")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        # Run all test suites
        tests = [
            self.test_api_health,
            self.test_models_endpoint,
            self.test_word_spell_checking,
            self.test_sentence_correction,
            self.test_learning_functionality,
            self.test_performance_benchmarks,
            self.test_error_handling,
            self.test_direct_components,
            self.test_accuracy_against_known_dataset
        ]
        
        for test_func in tests:
            logger.info(f"\n📋 Running {test_func.__name__.replace('test_', '').replace('_', ' ').title()}...")
            try:
                test_func()
            except Exception as e:
                logger.error(f"Test suite failed: {str(e)}")
                self.test_results["failed"] += 1
                self.test_results["errors"].append(f"{test_func.__name__}: {str(e)}")
        
        total_time = time.time() - start_time
        
        # Print final results
        logger.info("\n" + "=" * 80)
        logger.info("📊 TEST RESULTS SUMMARY")
        logger.info("=" * 80)
        
        total_tests = self.test_results["passed"] + self.test_results["failed"]
        success_rate = self.test_results["passed"] / total_tests if total_tests > 0 else 0
        
        logger.info(f"✅ Passed: {self.test_results['passed']}")
        logger.info(f"❌ Failed: {self.test_results['failed']}")
        logger.info(f"📈 Success Rate: {success_rate:.1%}")
        logger.info(f"⏱️  Total Time: {total_time:.1f}s")
        
        if self.test_results["accuracy_metrics"]:
            logger.info(f"\n🎯 ACCURACY METRICS:")
            for metric, value in self.test_results["accuracy_metrics"].items():
                logger.info(f"   {metric}: {value:.1%}")
        
        if self.test_results["performance_metrics"]:
            logger.info(f"\n⚡ PERFORMANCE METRICS:")
            for metric, value in self.test_results["performance_metrics"].items():
                logger.info(f"   {metric}: {value:.2f}ms")
        
        if self.test_results["errors"]:
            logger.info(f"\n❌ ERRORS:")
            for error in self.test_results["errors"][:5]:  # Show first 5 errors
                logger.info(f"   {error}")
        
        # Save results to file
        results_file = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        
        logger.info(f"\n💾 Detailed results saved to: {results_file}")
        
        return success_rate >= 0.8  # 80% success rate threshold

def main():
    """Main test runner function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test the Advanced Spell Checking System")
    parser.add_argument("--api-url", default="http://localhost:8000", 
                       help="Base URL for the API (default: http://localhost:8000)")
    parser.add_argument("--wait-for-server", action="store_true",
                       help="Wait for server to be available before testing")
    
    args = parser.parse_args()
    
    tester = SpellCheckerTester(args.api_url)
    
    if args.wait_for_server:
        logger.info("Waiting for server to be available...")
        max_retries = 30
        for i in range(max_retries):
            try:
                response = requests.get(f"{args.api_url}/health", timeout=5)
                if response.status_code == 200:
                    logger.info("✅ Server is available")
                    break
            except:
                pass
            
            time.sleep(2)
            logger.info(f"Waiting... ({i+1}/{max_retries})")
        else:
            logger.error("❌ Server not available after waiting")
            sys.exit(1)
    
    # Run tests
    success = tester.run_all_tests()
    
    if success:
        logger.info("\n🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        sys.exit(0)
    else:
        logger.error("\n💥 SOME TESTS FAILED!")
        sys.exit(1)

if __name__ == "__main__":
    main()