# vLLM Benchmark

This repository contains scripts for benchmarking the performance of large language models (LLMs) served using vLLM. It's designed to test the scalability and performance of LLM deployments under various concurrency levels.

## Features

- Benchmark LLMs with different concurrency levels
- Measure key performance metrics:
  - Requests per second
  - Latency
  - Tokens per second
  - Time to first token
- Easy to run with customizable parameters
- Generates JSON output for further analysis or visualization

## Requirements

- Python 3.7+
- `openai` Python package
- `numpy` Python package

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/vllm-benchmark.git
   cd vllm-benchmark
   ```

2. Install the required packages:
   ```
   pip install openai numpy
   ```

## Usage

### Single Benchmark Run

To run a single benchmark:

```
python vllm_benchmark.py --num_requests 100 --concurrency 10 --output_tokens 100 --vllm_url "http://localhost:8000/v1" --api_key "your-api-key"
```

Parameters:
- `num_requests`: Total number of requests to make
- `concurrency`: Number of concurrent requests
- `output_tokens`: Number of tokens to generate per request
- `vllm_url`: URL of the vLLM server
- `api_key`: API key for the vLLM server
- `request_timeout`: (Optional) Timeout for each request in seconds (default: 30)

### Multiple Benchmark Runs

To run multiple benchmarks with different concurrency levels:

```
python run_benchmarks.py --vllm_url "http://localhost:8000/v1" --api_key "your-api-key"
```

This script will run benchmarks with concurrency levels of 1, 10, 50, and 100, and save the results to `benchmark_results.json`.

## Output

The benchmark results are saved in JSON format, containing detailed metrics for each run, including:

- Total requests and successful requests
- Requests per second
- Total output tokens
- Latency (average, p50, p95, p99)
- Tokens per second (average, p50, p95, p99)
- Time to first token (average, p50, p95, p99)

## Results

Please see the results directory for benchmarks on [Backprop](https://backprop.co) instances.

## Contributing

Contributions to improve the benchmarking scripts or add new features are welcome! Please feel free to submit pull requests or open issues for any bugs or feature requests.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
