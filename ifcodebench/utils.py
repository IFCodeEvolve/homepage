import ast
import io
import json
import multiprocessing
import os
import re
import sys
import textwrap
import tokenize
import traceback
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


ACTOR_PROMPT = textwrap.dedent(
    """\
    As a programming assistant, your task is to generate code snippets based on the user question and instructions given below:

    # Requirements

    - Make sure you follow the user instructions. If the instruction says to use a specific language or a specific method, use exactly that.
    - Your output should be a valid code snippet in the programming language indicated in the question or the instructions.
    - Remember to import any necessary libraries or modules if needed.
    - Remember to import Typing if the signature contains type hints.

    # Output Format

    The output should only be a valid code snippet without any explanations, comments, or text outside the code.

    # Problem

    {prompt}
    """
)

_CODE_RE = re.compile(r"```.*?\n(.*?)```", re.DOTALL)


def read_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_data_auto(path: str):
    low = path.lower()
    if low.endswith(".jsonl"):
        return read_jsonl(path)
    if low.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    raise ValueError(f"Unsupported file type: {path}")


def write_data_auto(path: str, data) -> None:
    low = path.lower()
    if low.endswith(".jsonl"):
        write_jsonl(path, data)
        return
    if low.endswith(".json"):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return
    raise ValueError(f"Unsupported file type: {path}")


def extract_code(text: str) -> str:
    if not text:
        return ""
    match = _CODE_RE.search(text)
    return (match.group(1) if match else text).strip()


def infer_programming_language(instruction: str | None) -> str:
    text = (instruction or "").lower()
    if "c++" in text:
        return "cpp"
    if "c#" in text or "csharp" in text:
        return "csharp"
    if "typescript" in text:
        return "typescript"
    if "javascript" in text:
        return "javascript"
    if " php" in text:
        return "php"
    if "shell" in text or "bash" in text:
        return "shell"
    if " java" in text and "javascript" not in text:
        return "java"
    return "python"


def build_user_prompt(row: dict) -> str:
    prompt = row["question_desc"] + "\n\n## Instruction\n" + row["instruction"]
    prompt += "\n\n## Example Input and Output\n" + row.get("example_input_and_output", "")
    function_signature = row.get("function_signature")
    if function_signature:
        prompt += "\n\n## Function Signature\n" + function_signature
    return ACTOR_PROMPT.format(prompt=prompt)


def compute_group_stats(items: list[dict]) -> dict:
    n = len(items) or 1
    acc = sum(1 for x in items if x["total"]["if_correct"]) / n
    instr_mean = sum(x["total"]["instruction_score"] for x in items) / n
    instr_all = sum(1 for x in items if x["total"]["if_instruction"]) / n
    all_correct = sum(1 for x in items if x["total"]["all_correct"]) / n
    return {
        "n": len(items),
        "unit_test_accuracy": acc,
        "instruction_pass_rate": instr_mean,
        "prompt_pass_rate": instr_all,
        "all_correct": all_correct,
    }


def _normalize_chat_content(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text" and item.get("text"):
                    chunks.append(item["text"])
                elif "content" in item and isinstance(item["content"], str):
                    chunks.append(item["content"])
        return "".join(chunks)
    return str(content)


def _post_chat_completion(
    *,
    api_base: str,
    api_key: str | None,
    model_name: str,
    messages: list[dict],
    max_tokens: int,
    extra_body: dict | None,
    timeout: float,
) -> dict:
    url = api_base.rstrip("/") + "/chat/completions"
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if extra_body:
        payload.update(extra_body)

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def run_inference(
    *,
    model_name: str,
    input_path: str,
    output_path: str,
    repeats: int = 1,
    concurrency: int | None = None,
    max_tokens: int = 2048,
    api_base: str | None = None,
    api_key: str | None = None,
    extra_body: dict | None = None,
    timeout: float = 600.0,
) -> None:
    rows = read_data_auto(input_path)
    requests = []
    for _ in range(int(repeats)):
        requests.extend(dict(row) for row in rows)

    resolved_api_base = api_base or os.environ.get("OPENAI_BASE_URL") or os.environ.get("VLLM_BASE_URL")
    if not resolved_api_base:
        raise ValueError(
            "Missing API base URL. Pass --api-base or set OPENAI_BASE_URL/VLLM_BASE_URL."
        )

    worker_count = concurrency or 1
    results: list[dict | None] = [None] * len(requests)

    def _one(idx: int, row: dict) -> tuple[int, dict]:
        try:
            raw = _post_chat_completion(
                api_base=resolved_api_base,
                api_key=api_key or os.environ.get("OPENAI_API_KEY"),
                model_name=model_name,
                messages=row["prompt"],
                max_tokens=max_tokens,
                extra_body=extra_body,
                timeout=timeout,
            )
            choice = (raw.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            content = _normalize_chat_content(message.get("content", ""))
            out = dict(row)
            out.update(
                {
                    "status": "success",
                    "response": content,
                    "raw_response": raw,
                }
            )
            return idx, out
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            return idx, dict(row) | {"status": "error", "response": "", "error": f"HTTP {exc.code}: {body}"}
        except Exception:
            return idx, dict(row) | {"status": "error", "response": "", "error": traceback.format_exc()}

    with ThreadPoolExecutor(max_workers=max(1, worker_count)) as executor:
        futures = [executor.submit(_one, idx, row) for idx, row in enumerate(requests)]
        for future in as_completed(futures):
            idx, result = future.result()
            results[idx] = result

    write_data_auto(output_path, results)


def _correctness_worker(code: str, unit_test: str, entry_point: str | None, queue) -> None:
    capture_out = io.StringIO()
    capture_err = io.StringIO()
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = capture_out
    sys.stderr = capture_err

    result = {"success": False, "error": None, "stdout": ""}
    try:
        global_env = {"__name__": "__main__", "__builtins__": __builtins__}
        full_script = f"{code}\n\n{unit_test}\n\n"
        if entry_point:
            full_script += f"{entry_point}()"
        exec(full_script, global_env)
        result["success"] = True
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        result["stdout"] = capture_out.getvalue()
        queue.put(result)


def run_check_correctness(code: str, unit_test: str, entry_point: str = "check_correctness", timeout: float = 3.0) -> dict:
    queue = multiprocessing.Queue()
    proc = multiprocessing.Process(target=_correctness_worker, args=(code, unit_test, entry_point, queue))
    proc.start()
    proc.join(timeout)

    result = {"success": False, "reason": "", "stdout": "", "error": None}
    if proc.is_alive():
        proc.terminate()
        proc.join()
        result["reason"] = "Timeout"
        result["error"] = f"Code execution exceeded {timeout} seconds."
        return result

    if queue.empty():
        result["reason"] = "Process crashed silently (OOM or SegFault)"
        return result

    worker = queue.get()
    result["success"] = worker["success"]
    result["stdout"] = worker["stdout"]
    result["error"] = worker["error"]
    if not result["success"]:
        result["reason"] = (result["error"] or "Unknown Runtime Error").strip().split("\n")[-1]
    return result


class ConstraintValidator:
    def __init__(self, dynamic_functions=None):
        self.dynamic_functions = dynamic_functions or {}

    def validate(self, code: str, function_name: str, params: dict) -> bool:
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            raise ValueError("Invalid Python code syntax.") from exc

        if function_name in self.dynamic_functions:
            return self.dynamic_functions[function_name](tree, code_str=code, **params)
        if hasattr(self, function_name):
            return getattr(self, function_name)(tree, code_str=code, **params)
        raise NotImplementedError(f"Function {function_name} not implemented.")

    def check_variable_existence(self, tree, variable_name, should_define=True, **kwargs):
        assert should_define in ["should", "should not"]
        defined_vars = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        defined_vars.add(target.id)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                defined_vars.add(node.target.id)
        exists = variable_name in defined_vars
        return (not exists) if str(should_define).lower() == "should not" else exists

    def check_no_intermediate_variables(self, tree, **kwargs):
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        return False
                    if isinstance(target, (ast.Tuple, ast.List)):
                        for elt in target.elts:
                            if isinstance(elt, ast.Name):
                                return False
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                return False
            elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
                return False
            elif isinstance(node, ast.NamedExpr):
                return False
        return True

    def check_variable_naming_convention(self, tree, convention, **kwargs):
        assert convention in ["snake_case", "camelCase", "PascalCase"]
        vars_to_check = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        vars_to_check.add(target.id)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                vars_to_check.add(node.target.id)
        if not vars_to_check:
            return True
        patterns = {
            "snake_case": r"^[a-z_][a-z0-9_]*$",
            "camelCase": r"^[a-z][a-zA-Z0-9]*$",
            "PascalCase": r"^[A-Z][a-zA-Z0-9]*$",
        }
        pattern = patterns[convention]
        return all(re.match(pattern, var) for var in vars_to_check)

    def check_global_variable_existence(self, tree, variable_name, **kwargs):
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == variable_name:
                        return True
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == variable_name:
                return True
        for node in ast.walk(tree):
            if isinstance(node, ast.Global) and variable_name in node.names:
                return True
        return False

    def check_variable_initial_value(self, tree, variable_name, initial_value, **kwargs):
        def normalize_str(value):
            value = str(value).strip()
            if len(value) >= 2 and ((value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'"))):
                return value[1:-1]
            return value

        target_raw = str(initial_value).strip()
        target_normalized = normalize_str(target_raw)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if not isinstance(target, ast.Name) or target.id != variable_name:
                    continue
                try:
                    code_val_raw = ast.unparse(node.value).strip()
                except AttributeError:
                    if isinstance(node.value, ast.Constant):
                        code_val_raw = str(node.value.value)
                    elif isinstance(node.value, ast.Name):
                        code_val_raw = node.value.id
                    else:
                        continue
                if code_val_raw == target_raw or normalize_str(code_val_raw) == target_normalized:
                    return True
        return False

    def check_variable_name_length(self, tree, length, comparison="should not", length_comparison="exceed", **kwargs):
        assert comparison in ["should", "should not"]
        assert length_comparison in ["exceed", "be less than"]
        limit = int(length)
        vars_checked = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        vars_checked.append(target.id)
        if not vars_checked:
            return True
        for var in vars_checked:
            var_len = len(var)
            if length_comparison == "exceed":
                if comparison == "should not" and var_len > limit:
                    return False
                if comparison == "should" and var_len < limit:
                    return False
            else:
                if comparison == "should not" and var_len < limit:
                    return False
                if comparison == "should" and var_len > limit:
                    return False
        return True

    def check_variable_case(self, tree, case_type, **kwargs):
        assert case_type in ["lowercase", "uppercase"]
        vars_checked = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        vars_checked.append(target.id)
        if not vars_checked:
            return True
        for var in vars_checked:
            if case_type == "lowercase" and not var.islower():
                return False
            if case_type == "uppercase" and not var.isupper():
                return False
        return True

    def check_variable_prefix(self, tree, prefix, **kwargs):
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and not target.id.startswith(prefix):
                        return False
        return True

    def check_variable_suffix(self, tree, suffix, **kwargs):
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and not target.id.endswith(suffix):
                        return False
        return True

    def check_loop_count(self, tree, count, loop_type, comparison, **kwargs):
        assert comparison in ["at least", "at most", "exactly"]
        assert loop_type in ["for", "while"]
        if loop_type == "for":
            actual = len([n for n in ast.walk(tree) if isinstance(n, (ast.For, ast.AsyncFor))])
        else:
            actual = len([n for n in ast.walk(tree) if isinstance(n, ast.While)])
        target = int(count)
        if comparison == "at least":
            return actual >= target
        if comparison == "at most":
            return actual <= target
        return actual == target

    def check_forbidden_loop_type(self, tree, loop_type, **kwargs):
        return self.check_loop_count(tree, 0, loop_type, "exactly")

    def check_conditional_count(self, tree, count, comparison, **kwargs):
        assert comparison in ["at least", "at most", "exactly"]
        actual = len([n for n in ast.walk(tree) if isinstance(n, ast.If)])
        target = int(count)
        if comparison == "at least":
            return actual >= target
        if comparison == "at most":
            return actual <= target
        return actual == target

    def check_switch_statement_existence(self, tree, **kwargs):
        return any(hasattr(ast, "Match") and isinstance(node, ast.Match) for node in ast.walk(tree))

    def check_lines_of_code(self, tree, number, comparison, code_str="", **kwargs):
        assert comparison in ["at least", "at most", "exactly"]

        def meets_requirement(actual: int, target: int) -> bool:
            if comparison == "at least":
                return actual >= target
            if comparison == "at most":
                return actual <= target
            return actual == target

        target = int(number)
        lines = code_str.splitlines()
        total_non_empty = len([line for line in lines if line.strip()])
        if meets_requirement(total_non_empty, target):
            return True

        funcs = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        if len(funcs) != 1 or not funcs[0].body:
            return False
        func = funcs[0]
        body_start = func.body[0].lineno
        body_end = getattr(func, "end_lineno", body_start)
        body_lines = lines[body_start - 1 : body_end]
        body_non_empty = len([line for line in body_lines if line.strip()])
        return meets_requirement(body_non_empty, target)

    def check_syntactic_sugar_usage(self, tree, construct_type, should_use="must", **kwargs):
        assert construct_type in [
            "list comprehension",
            "ternary operator",
            "slicing",
            "extended slicing",
            "generator expression",
            "lambda function",
        ]
        assert should_use in ["must", "must not"]
        found = False
        for node in ast.walk(tree):
            if construct_type == "list comprehension" and isinstance(node, ast.ListComp):
                found = True
                break
            if construct_type == "ternary operator" and isinstance(node, ast.IfExp):
                found = True
                break
            if construct_type in {"slicing", "extended slicing"} and isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Slice):
                if construct_type == "slicing" or node.slice.step is not None:
                    found = True
                    break
            if construct_type == "generator expression" and isinstance(node, ast.GeneratorExp):
                found = True
                break
            if construct_type == "lambda function" and isinstance(node, ast.Lambda):
                found = True
                break
        return found == (should_use == "must")

    def check_inline_construct_count(self, tree, construct_type, count, comparison, **kwargs):
        assert comparison in ["at least", "at most", "exactly"]
        actual = 0
        for node in ast.walk(tree):
            if construct_type == "list comprehension" and isinstance(node, ast.ListComp):
                actual += 1
            elif construct_type == "ternary operator" and isinstance(node, ast.IfExp):
                actual += 1
            elif construct_type in {"slicing", "extended slicing"} and isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Slice):
                if construct_type == "slicing" or node.slice.step is not None:
                    actual += 1
            elif construct_type == "generator expression" and isinstance(node, ast.GeneratorExp):
                actual += 1
            elif construct_type == "lambda function" and isinstance(node, ast.Lambda):
                actual += 1
        target = int(count)
        if comparison == "at least":
            return actual >= target
        if comparison == "at most":
            return actual <= target
        return actual == target

    def check_function_definition(self, tree, function_name, **kwargs):
        return any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name
            for node in ast.walk(tree)
        )

    def check_function_parameter_count(self, tree, function_name, num, **kwargs):
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
                return len(node.args.args) == int(num)
        return False

    def check_recursion_usage(self, tree, **kwargs):
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            func_name = node.name
            for sub_node in ast.walk(node):
                if isinstance(sub_node, ast.Call) and isinstance(sub_node.func, ast.Name) and sub_node.func.id == func_name:
                    return True
        return False

    def check_forbidden_builtins(self, tree, forbidden_functions, **kwargs):
        forbidden = [item.strip() for item in forbidden_functions.split(",")] if isinstance(forbidden_functions, str) else forbidden_functions
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in forbidden:
                return False
        return True

    def check_mandatory_builtins(self, tree, required_functions, **kwargs):
        required = set(item.strip() for item in required_functions.split(",")) if isinstance(required_functions, str) else set(required_functions)
        called_funcs = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name):
                called_funcs.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                called_funcs.add(node.func.attr)
                try:
                    called_funcs.add(ast.unparse(node.func))
                except AttributeError:
                    pass
        for req in required:
            if req in called_funcs or req.split(".")[-1] in called_funcs:
                continue
            return False
        return True

    def check_data_structure_usage(self, tree, data_structure, must_use="must", **kwargs):
        assert data_structure in ["dict", "set", "list", "tuple"]
        found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == data_structure:
                found = True
            if data_structure == "list" and isinstance(node, (ast.List, ast.ListComp)):
                found = True
            if data_structure == "set" and isinstance(node, (ast.Set, ast.SetComp)):
                found = True
            if data_structure == "dict" and isinstance(node, (ast.Dict, ast.DictComp)):
                found = True
            if data_structure == "tuple" and isinstance(node, ast.Tuple):
                found = True
        return found == (str(must_use).lower() == "must")

    def check_interface_existence(self, tree, interface_name, **kwargs):
        return any(isinstance(node, ast.ClassDef) and node.name == interface_name for node in ast.walk(tree))

    def check_type_annotation_coverage(self, tree, **kwargs):
        total_args = 0
        annotated_args = 0
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for arg in node.args.args:
                if arg.arg == "self":
                    continue
                total_args += 1
                if arg.annotation:
                    annotated_args += 1
        if total_args == 0:
            return True
        return (annotated_args / total_args) >= 0.5

    def check_library_import(self, tree, library_name, should_import="must", **kwargs):
        assert should_import in ["must", "must not"]
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        exists = library_name in imported
        return exists == (str(should_import).lower() == "must")

    def check_library_no_import(self, tree, **kwargs):
        return not any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(tree))

    def check_comment_count(self, tree, count, comparison, code_str="", **kwargs):
        assert comparison in ["at least", "at most", "exactly"]
        comment_lines = set()
        try:
            tokens = tokenize.tokenize(io.BytesIO(code_str.encode("utf-8")).readline)
            for token in tokens:
                if token.type == tokenize.COMMENT:
                    comment_lines.add(token.start[0])
        except tokenize.TokenError:
            pass

        for node in ast.walk(tree):
            if not isinstance(node, ast.Expr) or not isinstance(node.value, (ast.Constant, ast.Str)):
                continue
            value = node.value.value if isinstance(node.value, ast.Constant) else node.value.s
            if not isinstance(value, str):
                continue
            start_line = getattr(node, "lineno", -1)
            end_line = getattr(node, "end_lineno", -1)
            if start_line != -1 and end_line != -1:
                for line_no in range(start_line, end_line + 1):
                    comment_lines.add(line_no)

        actual = len(comment_lines)
        target = int(count)
        if comparison == "at least":
            return actual >= target
        if comparison == "at most":
            return actual <= target
        return actual == target

    def check_comment_language(self, tree, language, code_str="", **kwargs):
        assert language in ["zh", "en"]
        comments = re.findall(r"#\s*(.*)", code_str)
        combined = " ".join(comments)
        has_chinese = bool(re.search(r"[\u4e00-\u9fff]", combined))
        if language == "zh":
            return has_chinese
        return (not has_chinese) and bool(combined)


def _build_dynamic_functions(constraints: list[dict]) -> dict[str, object]:
    custom_funcs: dict[str, object] = {}
    for item in constraints:
        impl = item.get("impl")
        func_name = item.get("function")
        if not impl or not func_name:
            continue
        scope = {"ast": ast, "re": re}
        exec(impl, scope, scope)
        if func_name not in scope:
            raise ValueError(f"Dynamic checker {func_name} was not defined by impl.")
        custom_funcs[func_name] = scope[func_name]
    return custom_funcs


def _instruction_worker(mutation, response_input: str, queue) -> None:
    result = {"score": None, "error": None, "success": False}
    try:
        constraints = []
        for group in mutation or []:
            for item in group:
                if not item.get("non_checkable", False):
                    constraints.append(item)
        if not constraints:
            result["score"] = 1.0
            result["success"] = True
            return

        validator = ConstraintValidator(dynamic_functions=_build_dynamic_functions(constraints))
        code = extract_code(response_input)
        passed = 0
        for item in constraints:
            params = item.get("params", {})
            if validator.validate(code, item["function"], params):
                passed += 1
        result["score"] = passed / len(constraints)
        result["success"] = True
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        queue.put(result)


def run_check_instruction(mutation, response_input: str, timeout: float = 5.0) -> dict:
    queue = multiprocessing.Queue()
    proc = multiprocessing.Process(target=_instruction_worker, args=(mutation, response_input, queue))
    proc.start()
    proc.join(timeout)

    result = {"result": None, "error": None, "success": False}
    if proc.is_alive():
        proc.terminate()
        proc.join()
        result["error"] = f"Validator execution exceeded {timeout} seconds."
        return result

    if queue.empty():
        result["error"] = "Validator process crashed silently."
        return result

    worker = queue.get()
    result["result"] = worker["score"]
    result["error"] = worker["error"]
    result["success"] = worker["success"]
    return result
