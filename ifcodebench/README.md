# IFCodeBench

IFCodeBench is a benchmark for instruction-following code generation.

Each example pairs a programming problem (`question_desc`) with an additional natural-language constraint (`instruction`). A model is asked to generate code that both:

- solves the problem, and
- satisfies the extra instruction constraints.

Evaluation has two parts:

- Functional correctness: run `check_correctness`.
- Instruction compliance: evaluate the structured `mutation` constraints and return a score in `[0, 1]`.

## Files

- `benchmark.jsonl`: benchmark data.
- `run_ifcodebench.py`: inference + evaluation entrypoint.
- `utils.py`: minimal local utilities for prompt construction, inference, and evaluation.

## Data format

Each line in [benchmark.jsonl](benchmark.jsonl) is one JSON object with the following fields:

- `question_desc` (string): problem statement.
- `instruction` (string): extra code-generation constraints.
- `function_signature` (string): expected function signature.
- `example_input_and_output` (string): informal examples.
- `lg` (string): language tag such as `en` or `zh`.
- `mutation` (list): structured constraint specification used for instruction evaluation.
- `check_correctness` (string): unit-test code that defines `check_correctness()`.
- `programming_language` (string): target language. In the current release, all examples are `python`.

The open benchmark intentionally omits several internal or answer-leaking fields from earlier versions, including ground-truth solutions and generated `check_instruction` scripts.

## Prompt construction

For each example, we build a single user message from:

- `question_desc`
- `instruction`
- `example_input_and_output`
- `function_signature` (if present)

It is then wrapped with a lightweight actor prompt template in `utils.py`.

## Running the benchmark

The runner uses an OpenAI-compatible chat-completions endpoint. You can either:

- pass `--api-base` and optionally `--api-key`, or
- set `OPENAI_BASE_URL` / `VLLM_BASE_URL` and `OPENAI_API_KEY`.

Example:

```bash
python ifcodebench/run_ifcodebench.py \
  --benchmark-path ifcodebench/benchmark.jsonl \
  --model-name Qwen2.5-Coder-7B-Instruct \
  --api-base http://127.0.0.1:8000/v1 \
  --output-dir outputs/ifcodebench_demo \
  --max-tokens 2048 \
  --repeats 1 \
  --concurrency 4
```

If `predictions.jsonl` already exists under `--output-dir`, the script will skip inference and run evaluation only.

## Outputs

The runner writes:

- `benchmark_prepared.jsonl`: benchmark rows used for the run.
- `inference_input.jsonl`: chat-formatted prompts.
- `predictions.jsonl`: raw model outputs.
- `predictions_aggregated.jsonl`: grouped completions per example.
- `eval_results_log.jsonl`: per-example evaluation details.
- `summary.json`: aggregated metrics.
