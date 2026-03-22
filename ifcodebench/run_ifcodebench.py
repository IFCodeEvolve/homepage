import argparse
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ifcodebench.utils import (
    build_user_prompt,
    compute_group_stats,
    extract_code,
    infer_programming_language,
    read_data_auto,
    run_check_correctness,
    run_check_instruction,
    run_inference,
    write_data_auto,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-path", type=str, default=os.path.join(ROOT, "ifcodebench", "benchmark.jsonl"))
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--use-reasoning-mode", action="store_true")
    parser.add_argument("--reasoning-effort", type=str, default="low")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--api-base", type=str, default=None)
    parser.add_argument("--api-key", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    rows = read_data_auto(args.benchmark_path)
    if args.limit is not None:
        rows = rows[: args.limit]

    for row in rows:
        row.setdefault("programming_language", infer_programming_language(row.get("instruction")))

    prepared_path = os.path.join(args.output_dir, "benchmark_prepared.jsonl")
    write_data_auto(prepared_path, rows)

    infer_input_path = os.path.join(args.output_dir, "inference_input.jsonl")
    if not os.path.exists(infer_input_path):
        prompts = [{"prompt": [{"role": "user", "content": build_user_prompt(row)}]} for row in rows]
        write_data_auto(infer_input_path, prompts)

    predictions_path = os.path.join(args.output_dir, "predictions.jsonl")
    if not os.path.exists(predictions_path):
        extra_body = {"thinking": {"type": "enabled" if args.use_reasoning_mode else "disabled"}}
        if args.use_reasoning_mode:
            extra_body["reasoning_effort"] = args.reasoning_effort
        run_inference(
            model_name=args.model_name,
            input_path=infer_input_path,
            output_path=predictions_path,
            repeats=args.repeats,
            concurrency=args.concurrency,
            max_tokens=args.max_tokens,
            api_base=args.api_base,
            api_key=args.api_key,
            extra_body=extra_body,
        )

    raw_predictions = read_data_auto(predictions_path)
    num_examples = len(rows)
    expected_predictions = num_examples * int(args.repeats)
    if len(raw_predictions) != expected_predictions:
        raise RuntimeError(
            f"predictions size mismatch: got {len(raw_predictions)}, expected {expected_predictions}"
        )

    aggregated_path = os.path.join(args.output_dir, "predictions_aggregated.jsonl")
    if not os.path.exists(aggregated_path):
        aggregated = []
        for index in range(num_examples):
            solutions = []
            for repeat in range(int(args.repeats)):
                response = raw_predictions[repeat * num_examples + index].get("response") or ""
                solutions.append({"response": response, "thinking": None})
            aggregated.append({"prompt": raw_predictions[index].get("prompt"), "solutions": solutions})
        write_data_auto(aggregated_path, aggregated)
    else:
        aggregated = read_data_auto(aggregated_path)

    eval_path = os.path.join(args.output_dir, "eval_results_log.jsonl")
    if not os.path.exists(eval_path):
        eval_rows = []
        start_time = time.time()
        for index, (row, prediction_group) in enumerate(zip(rows, aggregated), start=1):
            per_solution = {}
            best_instruction_score = 0.0
            any_correct = False

            for solution_index, solution in enumerate(prediction_group["solutions"]):
                response = solution.get("response") or ""
                code = extract_code(response)
                correctness = run_check_correctness(code, row["check_correctness"], entry_point="check_correctness")
                any_correct = any_correct or bool(correctness.get("success"))

                instruction = run_check_instruction(row.get("mutation"), response)
                instruction_score = float(instruction.get("result") or 0.0) if instruction.get("success") else 0.0
                best_instruction_score = max(best_instruction_score, instruction_score)

                per_solution[str(solution_index)] = {
                    "if_correct": bool(correctness.get("success")),
                    "if_instruction": instruction_score == 1.0,
                    "instruction_score": instruction_score,
                    "if_correct_stdout": correctness.get("stdout", ""),
                    "if_correct_error": correctness.get("error"),
                    "if_instruction_error": instruction.get("error"),
                }

            total = {
                "if_correct": any_correct,
                "instruction_score": best_instruction_score,
                "if_instruction": best_instruction_score == 1.0,
                "all_correct": any_correct and best_instruction_score == 1.0,
                "if_correct_rate": sum(1 for item in per_solution.values() if item["if_correct"]) / max(1, len(per_solution)),
                "if_instruction_rate": sum(item["instruction_score"] for item in per_solution.values()) / max(1, len(per_solution)),
            }

            eval_rows.append(
                {
                    "lg": row.get("lg"),
                    "programming_language": row.get("programming_language"),
                    "eval_results": per_solution,
                    "total": total,
                }
            )
            if index % 50 == 0:
                print(f"[eval] {index}/{num_examples} elapsed={time.time() - start_time:.1f}s")

        write_data_auto(eval_path, eval_rows)
    else:
        eval_rows = read_data_auto(eval_path)

    groups: dict[str, list[dict]] = {}
    for item in eval_rows:
        groups.setdefault("all", []).append(item)
        if item.get("programming_language"):
            groups.setdefault(f"pl:{item['programming_language']}", []).append(item)
        if item.get("lg"):
            groups.setdefault(f"lg:{item['lg']}", []).append(item)
        if item.get("lg") and item.get("programming_language"):
            groups.setdefault(f"lg_pl:{item['lg']}_{item['programming_language']}", []).append(item)

    summary = {
        "benchmark_path": args.benchmark_path,
        "model_name": args.model_name,
        "repeats": int(args.repeats),
        "max_tokens": int(args.max_tokens),
        "groups": {key: compute_group_stats(value) for key, value in groups.items()},
    }
    summary_path = os.path.join(args.output_dir, "summary.json")
    write_data_auto(summary_path, summary)
    print(summary_path)


if __name__ == "__main__":
    main()
