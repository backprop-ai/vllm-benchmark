import asyncio
import time
import numpy as np
from openai import AsyncOpenAI
import logging
import argparse
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constant prompt for all requests
PROMPT = "Explain the concept of artificial intelligence step by step."

async def process_stream(stream):
    first_token_time = None
    total_tokens = 0
    async for chunk in stream:
        if first_token_time is None:
            first_token_time = time.time()
        if chunk.choices[0].delta.content:
            total_tokens += 1
        if chunk.choices[0].finish_reason is not None:
            break
    return first_token_time, total_tokens

async def make_request(client, output_tokens, request_timeout):
    start_time = time.time()
    try:
        stream = await client.chat.completions.create(
            model="NousResearch/Meta-Llama-3.1-8B-Instruct",
            messages=[
                {"role": "user", "content": PROMPT}
            ],
            max_tokens=output_tokens,
            stream=True
        )
        first_token_time, total_tokens = await asyncio.wait_for(process_stream(stream), timeout=request_timeout)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        ttft = first_token_time - start_time if first_token_time else None
        tokens_per_second = total_tokens / elapsed_time if elapsed_time > 0 else 0
        return total_tokens, elapsed_time, tokens_per_second, ttft

    except asyncio.TimeoutError:
        logging.warning(f"Request timed out after {request_timeout} seconds")
        return None
    except Exception as e:
        logging.error(f"Error during request: {str(e)}")
        return None

async def worker(client, semaphore, queue, results, output_tokens, request_timeout):
    while True:
        async with semaphore:
            task_id = await queue.get()
            if task_id is None:
                queue.task_done()
                break
            logging.info(f"Starting request {task_id}")
            result = await make_request(client, output_tokens, request_timeout)
            if result:
                results.append(result)
            else:
                logging.warning(f"Request {task_id} failed")
            queue.task_done()
            logging.info(f"Finished request {task_id}")

def calculate_percentile(values, percentile, reverse=False):
    if not values:
        return None
    if reverse:
        return np.percentile(values, 100 - percentile)
    return np.percentile(values, percentile)

async def run_benchmark(num_requests, concurrency, request_timeout, output_tokens, vllm_url, api_key):
    client = AsyncOpenAI(base_url=vllm_url, api_key=api_key)
    semaphore = asyncio.Semaphore(concurrency)
    queue = asyncio.Queue()
    results = []

    # Add tasks to the queue
    for i in range(num_requests):
        await queue.put(i)
    
    # Add sentinel values to stop workers
    for _ in range(concurrency):
        await queue.put(None)

    # Create worker tasks
    workers = [asyncio.create_task(worker(client, semaphore, queue, results, output_tokens, request_timeout)) for _ in range(concurrency)]

    start_time = time.time()
    
    # Wait for all tasks to complete
    await queue.join()
    await asyncio.gather(*workers)

    end_time = time.time()

    # Calculate metrics
    total_elapsed_time = end_time - start_time
    total_tokens = sum(tokens for tokens, _, _, _ in results if tokens is not None)
    latencies = [elapsed_time for _, elapsed_time, _, _ in results if elapsed_time is not None]
    tokens_per_second_list = [tps for _, _, tps, _ in results if tps is not None]
    ttft_list = [ttft for _, _, _, ttft in results if ttft is not None]

    successful_requests = len(results)
    requests_per_second = successful_requests / total_elapsed_time if total_elapsed_time > 0 else 0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    avg_tokens_per_second = sum(tokens_per_second_list) / len(tokens_per_second_list) if tokens_per_second_list else 0
    avg_ttft = sum(ttft_list) / len(ttft_list) if ttft_list else 0
    
    # Calculate percentiles
    percentiles = [50, 95, 99]
    latency_percentiles = [calculate_percentile(latencies, p) for p in percentiles]
    tps_percentiles = [calculate_percentile(tokens_per_second_list, p, reverse=True) for p in percentiles]
    ttft_percentiles = [calculate_percentile(ttft_list, p) for p in percentiles]
    
    return {
        "total_requests": num_requests,
        "successful_requests": successful_requests,
        "concurrency": concurrency,
        "request_timeout": request_timeout,
        "max_output_tokens": output_tokens,
        "total_time": total_elapsed_time,
        "requests_per_second": requests_per_second,
        "total_output_tokens": total_tokens,
        "latency": {
            "average": avg_latency,
            "p50": latency_percentiles[0],
            "p95": latency_percentiles[1],
            "p99": latency_percentiles[2]
        },
        "tokens_per_second": {
            "average": avg_tokens_per_second,
            "p50": tps_percentiles[0],
            "p95": tps_percentiles[1],
            "p99": tps_percentiles[2]
        },
        "time_to_first_token": {
            "average": avg_ttft,
            "p50": ttft_percentiles[0],
            "p95": ttft_percentiles[1],
            "p99": ttft_percentiles[2]
        }
    }

def print_results(results):
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark LLaMA-3 model with vLLM")
    parser.add_argument("--num_requests", type=int, required=True, help="Number of requests to make")
    parser.add_argument("--concurrency", type=int, required=True, help="Number of concurrent requests")
    parser.add_argument("--request_timeout", type=int, default=30, help="Timeout for each request in seconds (default: 30)")
    parser.add_argument("--output_tokens", type=int, default=50, help="Number of output tokens (default: 50)")
    parser.add_argument("--vllm_url", type=str, required=True, help="URL of the vLLM server")
    parser.add_argument("--api_key", type=str, required=True, help="API key for vLLM server")
    args = parser.parse_args()

    results = asyncio.run(run_benchmark(args.num_requests, args.concurrency, args.request_timeout, args.output_tokens, args.vllm_url, args.api_key))
    print_results(results)
else:
    # When imported as a module, provide the run_benchmark function
    __all__ = ['run_benchmark']
