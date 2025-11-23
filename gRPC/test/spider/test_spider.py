"""
Spider数据集测试脚本
用于测试和比较 generate 和 generate0 两个函数在Text-to-SQL任务上的性能
"""

import sys
import os
import argparse
import time
import torch
import grpc
from tqdm import tqdm
import json
from typing import Dict, List

# 添加gRPC目录到路径
grpc_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, grpc_dir)

from speculative import *
from transformers import AutoTokenizer, AutoModelForCausalLM
import sd_pb2
import sd_pb2_grpc
from client import UAVNode, generate, generate0
from test.spider import (
    load_mini_spider, 
    format_spider_prompt, 
    extract_sql_query,
    calculate_exact_match
)


def evaluate_spider(
    uav_node: UAVNode,
    stub: sd_pb2_grpc.SDVerifyStub,
    tokenizer: AutoTokenizer,
    args: argparse.Namespace,
    generate_func,
    func_name: str,
    n_samples: int = 20
) -> Dict:
    """
    在Spider数据集上评估指定的生成函数
    
    Args:
        uav_node: UAV节点
        stub: gRPC stub
        tokenizer: tokenizer
        args: 参数
        generate_func: 生成函数 (generate 或 generate0)
        func_name: 函数名称
        n_samples: 样本数量
    
    Returns:
        包含评估结果的字典
    """
    print(f"\n{'='*60}")
    print(f"Testing function: {func_name}")
    print(f"{'='*60}\n")
    
    # 加载数据集
    dataset = load_mini_spider(n_samples=n_samples, seed=args.seed)
    
    # 统计指标
    total_exact_match = 0.0
    total_time = 0.0
    total_tokens = 0
    results = []
    
    # 遍历数据集
    for idx, item in enumerate(tqdm(dataset, desc=f"Spider Test ({func_name})")):
        # 格式化prompt
        prompt = format_spider_prompt(item)
        gold_sql = item.get("query", "")
        
        # Tokenize
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        
        print(f"\n--- Sample {idx+1}/{n_samples} ---")
        print(f"Question: {item.get('question', '')}")
        print(f"DB: {item.get('db_id', '')}")
        print(f"Gold SQL: {gold_sql}")
        
        # 生成
        start_time = time.time()
        try:
            output_text = generate_func(uav_node, stub, input_ids, tokenizer, args)
        except Exception as e:
            print(f"Error during generation: {e}")
            continue
        
        gen_time = time.time() - start_time
        
        # 提取SQL
        pred_sql = extract_sql_query(output_text)
        
        # 计算匹配分数
        exact_match = calculate_exact_match(pred_sql, gold_sql)
        
        print(f"Predicted SQL: {pred_sql}")
        print(f"Exact Match: {exact_match}")
        print(f"Generation Time: {gen_time:.2f}s")
        
        # 统计
        total_exact_match += exact_match
        total_time += gen_time
        
        # 计算生成的token数
        output_ids = tokenizer(output_text, return_tensors="pt").input_ids
        num_tokens = output_ids.shape[1] - input_ids.shape[1]
        total_tokens += num_tokens
        
        # 保存结果
        results.append({
            "idx": idx,
            "question": item.get("question", ""),
            "db_id": item.get("db_id", ""),
            "gold_sql": gold_sql,
            "pred_sql": pred_sql,
            "output_text": output_text,
            "exact_match": exact_match,
            "gen_time": gen_time,
            "num_tokens": num_tokens
        })
    
    # 计算平均指标
    avg_exact_match = total_exact_match / len(dataset)
    avg_time = total_time / len(dataset)
    avg_tokens = total_tokens / len(dataset)
    throughput = total_tokens / total_time if total_time > 0 else 0
    
    # 汇总结果
    summary = {
        "function": func_name,
        "n_samples": len(dataset),
        "avg_exact_match": avg_exact_match,
        "total_time": total_time,
        "avg_time_per_sample": avg_time,
        "total_tokens": total_tokens,
        "avg_tokens_per_sample": avg_tokens,
        "throughput": throughput,
        "results": results
    }
    
    # 打印汇总
    print(f"\n{'='*60}")
    print(f"Summary for {func_name}")
    print(f"{'='*60}")
    print(f"Samples: {len(dataset)}")
    print(f"Average Exact Match: {avg_exact_match:.3f}")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Avg Time per Sample: {avg_time:.2f}s")
    print(f"Total Tokens: {total_tokens}")
    print(f"Avg Tokens per Sample: {avg_tokens:.1f}")
    print(f"Throughput: {throughput:.2f} tokens/s")
    print(f"{'='*60}\n")
    
    return summary


def compare_functions(
    uav_node: UAVNode,
    stub: sd_pb2_grpc.SDVerifyStub,
    tokenizer: AutoTokenizer,
    args: argparse.Namespace,
    n_samples: int = 20
):
    """
    比较generate和generate0两个函数
    """
    print("\n" + "="*80)
    print("SPIDER DATASET COMPARISON: generate vs generate0")
    print("="*80)
    
    # 测试 generate0（基础版本）
    results_generate0 = evaluate_spider(
        uav_node, stub, tokenizer, args,
        generate_func=generate0,
        func_name="generate0 (Basic DSSD)",
        n_samples=n_samples
    )
    
    # 测试 generate（优化版本）
    results_generate = evaluate_spider(
        uav_node, stub, tokenizer, args,
        generate_func=generate,
        func_name="generate (Optimized DSSD)",
        n_samples=n_samples
    )
    
    # 比较结果
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    
    print(f"\n{'Metric':<30} {'generate0':<20} {'generate':<20} {'Improvement':<15}")
    print("-"*85)
    
    # Exact Match
    em0 = results_generate0["avg_exact_match"]
    em1 = results_generate["avg_exact_match"]
    em_diff = em1 - em0
    print(f"{'Avg Exact Match':<30} {em0:<20.3f} {em1:<20.3f} {em_diff:>+14.3f}")
    
    # Time
    time0 = results_generate0["total_time"]
    time1 = results_generate["total_time"]
    time_speedup = time0 / time1 if time1 > 0 else 0
    print(f"{'Total Time (s)':<30} {time0:<20.2f} {time1:<20.2f} {time_speedup:>13.2f}x")
    
    # Throughput
    tp0 = results_generate0["throughput"]
    tp1 = results_generate["throughput"]
    tp_speedup = tp1 / tp0 if tp0 > 0 else 0
    print(f"{'Throughput (tokens/s)':<30} {tp0:<20.2f} {tp1:<20.2f} {tp_speedup:>13.2f}x")
    
    print("-"*85)
    
    # 保存结果到JSON
    output_file = "spider_test_results.json"
    with open(output_file, "w") as f:
        json.dump({
            "generate0": results_generate0,
            "generate": results_generate,
            "comparison": {
                "exact_match_diff": em_diff,
                "time_speedup": time_speedup,
                "throughput_speedup": tp_speedup
            }
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return results_generate0, results_generate


def parse_arguments():
    parser = argparse.ArgumentParser(description='Spider Dataset Testing')
    
    # Model paths
    parser.add_argument('--draft_model_name', type=str, 
                        default="../../LLM/opt-125m",
                        help='Path to draft model')
    
    # Test parameters
    parser.add_argument('--n_samples', type=int, default=20,
                        help='Number of samples to test')
    parser.add_argument('--test_function', type=str, 
                        choices=['generate0', 'generate', 'both'],
                        default='both',
                        help='Which function to test')
    
    # Generation parameters
    parser.add_argument('--max_len', type=int, default=100,
                        help='Maximum generation length')
    parser.add_argument('--gamma', type=int, default=4,
                        help='Number of speculative tokens')
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--top_p', type=float, default=0)
    parser.add_argument('--seed', type=int, default=42)
    
    # Device
    parser.add_argument('--device', type=str, default='mps',
                        help='Device to use (mps, cuda, cpu)')
    
    # gRPC
    parser.add_argument('--server_addr', type=str, required=True,
                        help='gRPC server address, e.g., 127.0.0.1:50051')
    
    # Other
    parser.add_argument('--use_dist_summary', action='store_true',
                        help='Use compressed distribution')
    parser.add_argument('--no_cache', action='store_true',
                        help='Disable cache')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    
    # 设置设备
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # 加载模型和tokenizer
    print(f"Loading draft model from {args.draft_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.draft_model_name)
    draft_model = AutoModelForCausalLM.from_pretrained(args.draft_model_name)
    
    # 创建UAV节点
    uav_node = UAVNode(draft_model, device, args)
    
    # 连接gRPC服务器
    print(f"Connecting to gRPC server at {args.server_addr}...")
    with grpc.insecure_channel(args.server_addr) as channel:
        stub = sd_pb2_grpc.SDVerifyStub(channel)
        
        if args.test_function == 'both':
            # 比较两个函数
            compare_functions(uav_node, stub, tokenizer, args, args.n_samples)
        elif args.test_function == 'generate0':
            # 只测试generate0
            evaluate_spider(
                uav_node, stub, tokenizer, args,
                generate_func=generate0,
                func_name="generate0",
                n_samples=args.n_samples
            )
        elif args.test_function == 'generate':
            # 只测试generate
            evaluate_spider(
                uav_node, stub, tokenizer, args,
                generate_func=generate,
                func_name="generate",
                n_samples=args.n_samples
            )
    
    print("\n✓ Testing completed!")


if __name__ == "__main__":
    main()
