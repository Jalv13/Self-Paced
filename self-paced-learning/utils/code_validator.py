"""
AST-based code validation for quiz coding questions.
Provides syntax checking, structure analysis, and safe code execution
to deterministically grade student code before AI review.
"""

import ast
import subprocess
import tempfile
import os
import re
from typing import Dict, List, Optional, Any


class CodeValidator:
    """Validates student code using Python's AST module and safe execution."""

    # Map of construct names to AST node types
    CONSTRUCT_MAP = {
        "for_loop": ast.For,
        "while_loop": ast.While,
        "if_statement": ast.If,
        "function_def": ast.FunctionDef,
        "return": ast.Return,
        "break": ast.Break,
        "continue": ast.Continue,
        "list_comp": ast.ListComp,
        "dict_comp": ast.DictComp,
        "set_comp": ast.SetComp,
        "try_except": ast.Try,
        "class_def": ast.ClassDef,
        "import": ast.Import,
        "lambda": ast.Lambda,
        "assert": ast.Assert,
        "with_statement": ast.With,
    }

    # Built-in function names to detect via ast.Call
    BUILTIN_CALLS = {
        "print", "input", "len", "range", "int", "str", "float",
        "list", "dict", "set", "tuple", "sorted", "enumerate",
        "zip", "map", "filter", "sum", "min", "max", "abs",
        "type", "isinstance", "open", "append",
    }

    def validate_syntax(self, code: str) -> Dict[str, Any]:
        """
        Parse code into an AST to check for syntax errors.

        Args:
            code: Student's submitted Python code

        Returns:
            {
                "valid": bool,
                "error": str or None,
                "line": int or None,
                "offset": int or None
            }
        """
        if not code or not code.strip():
            return {
                "valid": False,
                "error": "No code submitted",
                "line": None,
                "offset": None,
            }

        # Strip common non-code artifacts (e.g., starter comment only)
        stripped = code.strip()
        lines = [l for l in stripped.split("\n") if l.strip() and not l.strip().startswith("#")]
        if not lines:
            return {
                "valid": False,
                "error": "No executable code found (only comments or blank lines)",
                "line": None,
                "offset": None,
            }

        try:
            ast.parse(code)
            return {"valid": True, "error": None, "line": None, "offset": None}
        except SyntaxError as e:
            return {
                "valid": False,
                "error": e.msg,
                "line": e.lineno,
                "offset": e.offset,
            }

    def analyze_structure(
        self, code: str, expected_constructs: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Walk the AST to detect which programming constructs are used.

        Args:
            code: Student's Python code
            expected_constructs: List of construct names to check for
                e.g., ["for_loop", "print", "function_def"]

        Returns:
            {
                "constructs_found": list of construct names found,
                "missing": list of expected constructs not found,
                "all_present": bool
            }
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return {
                "constructs_found": [],
                "missing": expected_constructs or [],
                "all_present": False,
            }

        found = set()

        for node in ast.walk(tree):
            # Check statement-level constructs
            for construct_name, node_type in self.CONSTRUCT_MAP.items():
                if isinstance(node, node_type):
                    found.add(construct_name)

            # Check function/method calls
            if isinstance(node, ast.Call):
                # Direct call: print(), len(), etc.
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    if func_name in self.BUILTIN_CALLS:
                        found.add(func_name)
                # Method call: list.append(), str.split(), etc.
                elif isinstance(node.func, ast.Attribute):
                    method_name = node.func.attr
                    if method_name in self.BUILTIN_CALLS:
                        found.add(method_name)

            # Check for augmented assignments (+=, -=, etc.)
            if isinstance(node, ast.AugAssign):
                found.add("augmented_assign")

            # Check for string formatting
            if isinstance(node, ast.JoinedStr):
                found.add("f_string")

        expected = expected_constructs or []
        missing = [c for c in expected if c not in found]

        return {
            "constructs_found": sorted(list(found)),
            "missing": missing,
            "all_present": len(missing) == 0,
        }

    def compare_output(
        self, code: str, expected_output: str, timeout: int = 5
    ) -> Dict[str, Any]:
        """
        Safely execute student code in a subprocess and compare output.

        Args:
            code: Student's Python code
            expected_output: Expected stdout output
            timeout: Maximum execution time in seconds

        Returns:
            {
                "executed": bool,
                "output": str (actual stdout),
                "error": str or None (stderr or timeout message),
                "matches": bool,
                "return_code": int
            }
        """
        if not code or not code.strip():
            return {
                "executed": False,
                "output": "",
                "error": "No code to execute",
                "matches": False,
                "return_code": -1,
            }

        temp_path = None
        try:
            # Write code to a temporary file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, encoding="utf-8"
            ) as f:
                f.write(code)
                temp_path = f.name

            # Execute with timeout and capture output
            result = subprocess.run(
                ["python3", temp_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                env={
                    **os.environ,
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONIOENCODING": "utf-8",
                },
            )

            actual_output = result.stdout.strip()
            expected_clean = expected_output.strip()

            # Normalize whitespace for comparison (collapse multiple spaces, normalize newlines)
            actual_normalized = re.sub(r" +", " ", actual_output)
            expected_normalized = re.sub(r" +", " ", expected_clean)

            matches = actual_normalized == expected_normalized

            return {
                "executed": True,
                "output": actual_output,
                "error": result.stderr.strip() if result.returncode != 0 else None,
                "matches": matches,
                "return_code": result.returncode,
            }

        except subprocess.TimeoutExpired:
            return {
                "executed": False,
                "output": "",
                "error": f"Code execution timed out after {timeout} seconds (possible infinite loop)",
                "matches": False,
                "return_code": -1,
            }
        except Exception as e:
            return {
                "executed": False,
                "output": "",
                "error": f"Execution error: {str(e)}",
                "matches": False,
                "return_code": -1,
            }
        finally:
            # Always clean up the temp file
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

    def full_validate(
        self,
        code: str,
        expected_output: Optional[str] = None,
        expected_constructs: Optional[List[str]] = None,
        sample_solution: Optional[str] = None,
        timeout: int = 5,
    ) -> Dict[str, Any]:
        """
        Run full validation pipeline: syntax → structure → output comparison.

        Returns a comprehensive result dict with grading recommendation.
        """
        result = {
            "syntax": None,
            "structure": None,
            "execution": None,
            "status": "Unknown",
            "is_correct": False,
            "feedback_details": "",
        }

        # Step 1: Syntax validation
        syntax = self.validate_syntax(code)
        result["syntax"] = syntax

        if not syntax["valid"]:
            result["status"] = "Incorrect (Syntax Error)"
            line_info = f" on line {syntax['line']}" if syntax["line"] else ""
            result["feedback_details"] = (
                f"Syntax Error{line_info}: {syntax['error']}"
            )
            return result

        # Step 2: Structure analysis (if expected constructs provided)
        if expected_constructs:
            structure = self.analyze_structure(code, expected_constructs)
            result["structure"] = structure

            if not structure["all_present"]:
                result["status"] = "Incorrect (Missing Constructs)"
                missing_str = ", ".join(structure["missing"])
                result["feedback_details"] = (
                    f"Missing required constructs: {missing_str}\n"
                    f"Found: {', '.join(structure['constructs_found'])}"
                )
                # Don't return yet — still try to execute

        # Step 3: Output comparison (if expected output provided)
        if expected_output:
            execution = self.compare_output(code, expected_output, timeout)
            result["execution"] = execution

            if execution.get("matches"):
                result["status"] = "Correct"
                result["is_correct"] = True
                result["feedback_details"] = "Output matches expected result."
            elif execution.get("error"):
                if "timed out" in execution["error"]:
                    result["status"] = "Incorrect (Timeout)"
                    result["feedback_details"] = execution["error"]
                else:
                    result["status"] = "Incorrect (Runtime Error)"
                    result["feedback_details"] = (
                        f"Runtime Error: {execution['error']}"
                    )
            else:
                result["status"] = "Incorrect (Wrong Output)"
                result["feedback_details"] = (
                    f"Expected Output:\n{expected_output.strip()}\n\n"
                    f"Your Output:\n{execution.get('output', '(no output)')}"
                )
        elif not expected_constructs:
            # No expected output and no expected constructs — needs AI review
            result["status"] = "For AI Review"
            result["feedback_details"] = (
                "Code has valid syntax. Sent to AI for detailed review."
            )

        # If structure check failed but output matched, still mark correct
        # (student may have used alternative approach)
        if result.get("is_correct") and result.get("structure") and not result["structure"]["all_present"]:
            result["feedback_details"] += (
                f"\nNote: Output is correct, but expected constructs "
                f"({', '.join(result['structure']['missing'])}) were not detected."
            )

        return result
