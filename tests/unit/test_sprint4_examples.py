import sys
import os
from lark import Lark, UnexpectedInput

# Add project root to path to allow importing the parser
project_root = "/home/ubuntu/omega_project"
sys.path.insert(0, os.path.join(project_root, "src"))

# Import the original parser module (assuming it contains the grammar and parser instance)
try:
    # Use the original parser which is the main one despite known issues
    from core.languages.metalang_parser import omega_grammar, OmegaMetaLangTransformer
    print("Imported original parser (metalang_parser.py) successfully.")
except ImportError as e:
    print(f"Error importing original parser: {e}")
    # Fallback to simplified if original is missing (should not happen in this branch)
    try:
        from core.languages.metalang_parser_simplified import omega_grammar, OmegaMetaLangSimplifiedTransformer as OmegaMetaLangTransformer
        print("Imported simplified parser (metalang_parser_simplified.py) as fallback.")
    except ImportError as e2:
        print(f"Error importing simplified parser as fallback: {e2}")
        sys.exit(1)

# List of example files to test
example_files = [
    "/home/ubuntu/omega_project/examples/sprint4/rl/cartpole_ppo.omega",
    "/home/ubuntu/omega_project/examples/sprint4/probability/bayesian_regression.omega",
    "/home/ubuntu/omega_project/examples/sprint4/causal/medical_diagnosis.omega",
    "/home/ubuntu/omega_project/examples/sprint4/guarantees/robust_classifier.omega",
]

# Initialize parser
try:
    # Ensure the correct transformer is used based on import success
    if 'OmegaMetaLangSimplifiedTransformer' in locals() and 'OmegaMetaLangTransformer' not in locals():
         parser = Lark(omega_grammar, start='omega_program', parser='lalr', transformer=OmegaMetaLangSimplifiedTransformer())
    else:
         parser = Lark(omega_grammar, start='omega_program', parser='lalr', transformer=OmegaMetaLangTransformer())
    print("Parser initialized successfully.")
except Exception as e:
    print(f"Error initializing Lark parser: {e}")
    sys.exit(1)


# Test each file
results = {}
for file_path in example_files:
    print(f"--- Testing: {os.path.basename(file_path)} ---")
    try:
        with open(file_path, 'r') as f:
            code = f.read()
        
        # Attempt to parse
        tree = parser.parse(code)
        print("Parsing successful!")
        # print("AST:", tree) # Optional: print AST if needed
        results[file_path] = {"status": "success", "error": None}
        
    except UnexpectedInput as e:
        print(f"Parsing failed: Unexpected input at line {e.line}, column {e.column}.")
        # Provide more context around the error
        context_lines = 3
        start_line = max(0, e.line - context_lines)
        end_line = e.line + context_lines -1
        code_lines = code.splitlines()
        context = "\n".join(f"{i+1: >4}: {line}" for i, line in enumerate(code_lines[start_line:end_line]))
        pointer = " " * (e.column + 5) + "^"
        print(f"Context:\n{context}\n{pointer}")
        print(f"Expected one of: {e.expected}")
        results[file_path] = {"status": "failed", "error": f"UnexpectedInput: line {e.line}, col {e.column}, expected {e.expected}"}
    except Exception as e:
        import traceback
        print(f"Parsing failed with unexpected error: {type(e).__name__}: {e}")
        # print(traceback.format_exc()) # Uncomment for full traceback if needed
        results[file_path] = {"status": "failed", "error": f"{type(e).__name__}: {e}"}
    print("-" * (len(os.path.basename(file_path)) + 14))

# Summary
print("\n--- Summary ---")
success_count = 0
for file, result in results.items():
    print(f"{os.path.basename(file)}: {result['status']}")
    if result['status'] == 'failed':
        print(f"  Error: {result['error']}")
    else:
        success_count += 1
print(f"\n{success_count}/{len(example_files)} examples parsed successfully.")

# Exit with non-zero code if any test failed
if success_count < len(example_files):
    sys.exit(1)

