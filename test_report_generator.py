import unittest
import os
import json
import tempfile
import shutil # Import shutil for directory removal
from io import BytesIO # Used to suppress matplotlib output in tests if needed

# Assuming report_generator.py is in the same directory or accessible via PYTHONPATH
from report_generator import ChecklistReportGenerator, MPL_COLOR_GRAY, MPL_COLOR_TEXT_PRIMARY

# Suppress matplotlib figure pop-ups during tests if they occur
# import matplotlib
# matplotlib.use('Agg') # Already set in report_generator.py, but good practice here too

# --- Sample Test Data ---

VALID_TEST_DATA = [
  {
    "check_item": {
      "id": "CHK001",
      "name": "Initial Setup Verification",
      "description": "Verify the initial system configuration.",
      "phase": "Setup",
      "branch_name": "Configuration",
      "branch_id": "B01"
    },
    "is_met": True,
    "reliability": 95,
    "needs_human_review": False,
    "analysis_details": "Configuration files match the expected template.",
    "sources": ["/path/to/config.yaml", "System logs entry A"],
    "user_input": None
  },
  {
    "check_item": {
      "id": "CHK002",
      "name": "Dependency Check",
      "description": "Ensure all required dependencies are installed.",
      "phase": "Setup",
      "branch_name": "Dependencies",
      "branch_id": "B02"
    },
    "is_met": False,
    "reliability": 75,
    "needs_human_review": True,
    "analysis_details": "Package 'libfoo' version mismatch. Found 1.2, expected 1.3+.",
    "sources": ["Package manager query output"],
    "user_input": "User provided info: Waiting for manual update of libfoo."
  },
  {
    "check_item": {
      "id": "CHK003",
      "name": "Network Connectivity",
      "description": "Test connection to external services.",
      "phase": "Setup",
      "branch_name": "Networking",
      "branch_id": "B03"
    },
    "is_met": True,
    "reliability": 88,
    "needs_human_review": False,
    "analysis_details": "Successfully pinged service X and Y.",
    "sources": ["Ping results log"],
    "user_input": "1" # Example of acceptance input
  },
    {
    "check_item": {
      "id": "CHK004",
      "name": "Security Scan",
      "description": "Run basic security vulnerability scan.",
      "phase": "Security",
      "branch_name": "Vulnerabilities",
      "branch_id": "B04"
    },
    "is_met": False,
    "reliability": 40,
    "needs_human_review": True,
    "analysis_details": "Scan tool reported low confidence potential issue.",
    "sources": ["scan_report.txt"],
    "user_input": "skip" # Example of acceptance input
  },
    {
    "check_item": {
      "id": "CHK005",
      "name": "Performance Baseline",
      "description": "Check baseline performance metrics.",
      "phase": "Performance", # Different phase
      "branch_name": "Configuration", # Same branch as CHK001
      "branch_id": "B01"
    },
    "is_met": True,
    "reliability": 65, # Different reliability bracket
    "needs_human_review": False,
    "analysis_details": "CPU and memory usage within normal range.",
    "sources": ["perf_metrics.csv"],
    "user_input": None
  }
]

ERROR_MESSAGE_DATA = [
  {"error": "Required input data file missing."}
]

EMPTY_LIST_DATA = []

INVALID_JSON_STRING = """
[
  {
    "check_item": {
      "id": "CHK001",
      "name": "Initial Setup" // Missing comma
      "description": "Verify config."
    },
    "is_met": true
  }
]
"""

# --- Test Class ---

class TestChecklistReportGenerator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up resources shared by all tests in the class (optional)."""
        # Define a specific directory for test outputs relative to the script location
        cls.test_output_dir = os.path.join(os.path.dirname(__file__), "test_report_output")
        # Remove the directory if it exists from a previous run
        if os.path.exists(cls.test_output_dir):
            shutil.rmtree(cls.test_output_dir)
        # Create the fresh directory
        os.makedirs(cls.test_output_dir)
        print(f"Created test output directory: {cls.test_output_dir}")

        # Keep a temporary directory just for input files
        cls.temp_input_dir = tempfile.TemporaryDirectory()
        cls.input_dir_path = cls.temp_input_dir.name

        # --- Create Input JSON files in the temporary input directory ---
        cls.valid_json_path = os.path.join(cls.input_dir_path, "valid_data.json")
        with open(cls.valid_json_path, 'w', encoding='utf-8') as f:
            json.dump(VALID_TEST_DATA, f, indent=2)

        cls.error_json_path = os.path.join(cls.input_dir_path, "error_data.json")
        with open(cls.error_json_path, 'w', encoding='utf-8') as f:
            json.dump(ERROR_MESSAGE_DATA, f, indent=2)

        cls.empty_json_path = os.path.join(cls.input_dir_path, "empty_data.json")
        with open(cls.empty_json_path, 'w', encoding='utf-8') as f:
            json.dump(EMPTY_LIST_DATA, f, indent=2)

        cls.invalid_json_path = os.path.join(cls.input_dir_path, "invalid_data.json")
        with open(cls.invalid_json_path, 'w', encoding='utf-8') as f:
            f.write(INVALID_JSON_STRING)


    @classmethod
    def tearDownClass(cls):
        """Clean up resources shared by all tests after all tests run."""
        # Clean up the temporary input directory
        cls.temp_input_dir.cleanup()
        print(f"Cleaned up temporary input directory.")
        # Optionally remove the test output directory after inspection
        # shutil.rmtree(cls.test_output_dir)
        # print(f"Removed test output directory: {cls.test_output_dir}")
        print(f"Test outputs left in: {cls.test_output_dir}") # Keep output for inspection

    def setUp(self):
        """Set up specifics for each test method."""
        # Define a standard output path for generated PDFs within the class output dir
        # Use the test method name to create unique output files per test
        test_method_name = self._testMethodName
        self.output_pdf_path = os.path.join(self.test_output_dir, f"{test_method_name}_report.pdf")


    def tearDown(self):
        """Clean up after each test method (if needed)."""
        # Individual file cleanup could happen here if needed,
        # but setUpClass/tearDownClass handles the main directories.
        pass

    # --- Initialization Tests ---

    def test_init_valid_data(self):
        """Test initialization with a valid JSON file."""
        # Test default output path creation (when output_pdf_path=None)
        # For this, we need to know the *input* path
        generator_default = ChecklistReportGenerator(self.valid_json_path) # No output path specified
        self.assertTrue(generator_default.has_valid_data)
        self.assertIsNone(generator_default.error_message)
        self.assertEqual(len(generator_default.results), len(VALID_TEST_DATA))
        self.assertEqual(generator_default.phase, VALID_TEST_DATA[0]['check_item']['phase'])
        # Check default output path creation relative to *runtime* location (might be project root or test dir)
        # The default logic creates ./output/ relative to where python is run
        expected_output_base = os.path.splitext(os.path.basename(self.valid_json_path))[0]
        expected_default_path = os.path.join("output", f"{expected_output_base}_report.pdf")
        # Use normpath for OS-agnostic comparison
        self.assertEqual(os.path.normpath(generator_default.output_pdf_path), os.path.normpath(expected_default_path))

    def test_init_valid_data_explicit_output(self):
        """Test initialization with a valid JSON file and explicit output path."""
        # Uses self.output_pdf_path set in setUp, pointing to test_report_output dir
        generator = ChecklistReportGenerator(self.valid_json_path, output_pdf_path=self.output_pdf_path)
        self.assertTrue(generator.has_valid_data)
        self.assertIsNone(generator.error_message)
        self.assertEqual(generator.output_pdf_path, self.output_pdf_path) # Check explicit path is used

    def test_init_invalid_json(self):
        """Test initialization with a malformed JSON file."""
        generator = ChecklistReportGenerator(self.invalid_json_path, output_pdf_path=self.output_pdf_path) # Specify output for consistency
        self.assertFalse(generator.has_valid_data)
        self.assertIn("Error decoding JSON", generator.error_message)
        self.assertEqual(generator.results, [])

    def test_init_empty_list(self):
        """Test initialization with an empty JSON list."""
        generator = ChecklistReportGenerator(self.empty_json_path, output_pdf_path=self.output_pdf_path)
        self.assertFalse(generator.has_valid_data)
        self.assertEqual(generator.error_message, "JSON file contains no valid check results.")
        self.assertEqual(generator.results, [])

    def test_init_error_message_data(self):
        """Test initialization with JSON containing only an error message."""
        generator = ChecklistReportGenerator(self.error_json_path, output_pdf_path=self.output_pdf_path)
        self.assertFalse(generator.has_valid_data)
        self.assertEqual(generator.error_message, ERROR_MESSAGE_DATA[0]['error'])
        self.assertEqual(generator.results, ERROR_MESSAGE_DATA)

    def test_init_nonexistent_file(self):
        """Test initialization with a non-existent JSON file path."""
        non_existent_path = os.path.join(self.input_dir_path, "does_not_exist.json")
        generator = ChecklistReportGenerator(non_existent_path, output_pdf_path=self.output_pdf_path)
        self.assertFalse(generator.has_valid_data)
        self.assertIn("Error loading or parsing JSON data", generator.error_message)
        self.assertIn("No such file or directory", generator.error_message)
        self.assertEqual(generator.results, [])

    # --- PDF Generation Tests ---

    def test_generate_pdf_valid_data(self):
        """Test generating a PDF from valid data."""
        # Uses self.output_pdf_path set in setUp
        generator = ChecklistReportGenerator(self.valid_json_path, output_pdf_path=self.output_pdf_path)
        generated_path = generator.generate_pdf()

        self.assertEqual(generated_path, self.output_pdf_path)
        self.assertTrue(os.path.exists(self.output_pdf_path))
        self.assertGreater(os.path.getsize(self.output_pdf_path), 100)

    def test_generate_pdf_error_data(self):
        """Test generating a PDF when initialized with error data."""
        # Uses self.output_pdf_path set in setUp
        generator = ChecklistReportGenerator(self.error_json_path, output_pdf_path=self.output_pdf_path)
        generated_path = generator.generate_pdf()

        self.assertEqual(generated_path, self.output_pdf_path)
        self.assertTrue(os.path.exists(self.output_pdf_path))
        self.assertGreater(os.path.getsize(self.output_pdf_path), 100)

    def test_generate_pdf_invalid_json_init(self):
        """Test generating PDF when initialized with invalid JSON (should produce error report)."""
        # Uses self.output_pdf_path set in setUp
        generator = ChecklistReportGenerator(self.invalid_json_path, output_pdf_path=self.output_pdf_path)
        generated_path = generator.generate_pdf()

        self.assertEqual(generated_path, self.output_pdf_path)
        self.assertTrue(os.path.exists(self.output_pdf_path))
        self.assertGreater(os.path.getsize(self.output_pdf_path), 100)

    def test_generate_pdf_build_exception(self):
        """Test handling of exceptions during PDF build (e.g., ReportLab error)."""
        # Uses self.output_pdf_path set in setUp
        generator = ChecklistReportGenerator(self.valid_json_path, output_pdf_path=self.output_pdf_path)

        # Mock the build method to raise an exception
        original_build = generator.generate_pdf.__globals__['SimpleDocTemplate'].build
        def mock_build(*args, **kwargs):
            raise Exception("Simulated ReportLab build error")

        # Apply the mock carefully using unittest.mock if possible, or direct patch:
        # Ideally use: @patch('report_generator.SimpleDocTemplate.build', mock_build) above the test method
        # Direct patch for simplicity here:
        generator.generate_pdf.__globals__['SimpleDocTemplate'].build = mock_build

        generated_path = generator.generate_pdf()

        # Restore original build method
        generator.generate_pdf.__globals__['SimpleDocTemplate'].build = original_build

        self.assertIsNone(generated_path)
        error_txt_path = self.output_pdf_path.replace(".pdf", "_error.txt")
        self.assertTrue(os.path.exists(error_txt_path))
        with open(error_txt_path, 'r', encoding='utf-8') as f:
            content = f.read()
            self.assertIn("Failed to generate PDF report", content)
            self.assertIn("Simulated ReportLab build error", content)


    # --- Component Creation Tests (Basic checks) ---

    def test_create_summary_table(self):
        """Test summary table creation."""
        generator = ChecklistReportGenerator(self.valid_json_path, output_pdf_path=self.output_pdf_path)
        table = generator.create_summary_table()
        self.assertIsNotNone(table)
        self.assertEqual(len(table._cellvalues), len(VALID_TEST_DATA) + 1)

        generator_invalid = ChecklistReportGenerator(self.empty_json_path, output_pdf_path=self.output_pdf_path)
        table_invalid = generator_invalid.create_summary_table()
        self.assertIsNone(table_invalid)

    def test_create_charts(self):
        """Test chart creation (checks if images are generated)."""
        generator = ChecklistReportGenerator(self.valid_json_path, output_pdf_path=self.output_pdf_path)
        charts = generator.create_charts()
        self.assertGreaterEqual(len(charts), 3)
        from reportlab.platypus import Image, Paragraph, Table
        self.assertIsInstance(charts[0], Image)
        self.assertIsInstance(charts[1], Image)
        self.assertIsInstance(charts[2], Image)
        self.assertIsInstance(charts[3], Paragraph)
        self.assertIsInstance(charts[5], Table)

        generator_invalid = ChecklistReportGenerator(self.empty_json_path, output_pdf_path=self.output_pdf_path)
        charts_invalid = generator_invalid.create_charts()
        self.assertEqual(charts_invalid, [])

    def test_create_detailed_sections(self):
        """Test detailed section creation."""
        generator = ChecklistReportGenerator(self.valid_json_path, output_pdf_path=self.output_pdf_path)
        sections = generator.create_detailed_sections()
        self.assertEqual(len(sections), len(VALID_TEST_DATA) * 2)
        from reportlab.platypus import KeepTogether, Spacer
        self.assertIsInstance(sections[0], KeepTogether)
        self.assertIsInstance(sections[1], Spacer)

        generator_invalid = ChecklistReportGenerator(self.empty_json_path, output_pdf_path=self.output_pdf_path)
        sections_invalid = generator_invalid.create_detailed_sections()
        self.assertEqual(sections_invalid, [])


if __name__ == '__main__':
     # Ensure the runner looks for tests in this module
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestChecklistReportGenerator))
    runner = unittest.TextTestRunner()
    runner.run(suite)

    # Previous simpler run command:
    # unittest.main(argv=['first-arg-is-ignored'], exit=False) 