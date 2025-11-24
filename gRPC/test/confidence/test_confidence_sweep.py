"""
批量测试不同confidence阈值下的吞吐量
测试confidence = 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6
生成max_len=1000的文段
"""

import argparse
import torch
import grpc
import time
import json
import csv
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

import sd_pb2
import sd_pb2_grpc
from client import UAVNode, generate


def save_results_to_json(results, output_file="confidence_sweep_results.json"):
    """保存结果到JSON文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Results saved to JSON: {output_file}")


def save_results_to_csv(results, output_file="confidence_sweep_results.csv"):
    """保存结果到CSV文件"""
    fieldnames = [
        'confidence_threshold',
        'throughput',
        'total_time',
        'total_tokens',
        'total_rounds',
        'avg_gamma',
        'acceptance_rate',
        'generated_text_length',
        'timestamp'
    ]
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results['experiments']:
            writer.writerow({
                'confidence_threshold': result['confidence_threshold'],
                'throughput': result['throughput'],
                'total_time': result['total_time'],
                'total_tokens': result['total_tokens'],
                'total_rounds': result['total_rounds'],
                'avg_gamma': result['avg_gamma'],
                'acceptance_rate': result['acceptance_rate'],
                'generated_text_length': len(result['generated_text']),
                'timestamp': result['timestamp']
            })
    
    print(f"✓ Results saved to CSV: {output_file}")


def print_summary_table(results):
    """打印结果汇总表"""
    print("\n" + "="*100)
    print("CONFIDENCE THRESHOLD SWEEP RESULTS")
    print("="*100)
    print(f"{'Confidence':<12} {'Throughput':<15} {'Total Time':<12} {'Tokens':<10} {'Rounds':<10} {'Avg γ':<10} {'Accept Rate':<12}")
    print("-"*100)
    
    for result in results['experiments']:
        print(f"{result['confidence_threshold']:<12.3f} "
              f"{result['throughput']:<15.2f} "
              f"{result['total_time']:<12.2f} "
              f"{result['total_tokens']:<10} "
              f"{result['total_rounds']:<10} "
              f"{result['avg_gamma']:<10.2f} "
              f"{result['acceptance_rate']:<12.3f}")
    
    print("="*100)


def test_single_confidence(
    uav_node,
    stub,
    tokenizer,
    args,
    confidence_threshold
):
    """测试单个confidence阈值"""
    
    print(f"\n{'='*80}")
    print(f"Testing confidence_threshold = {confidence_threshold}")
    print(f"{'='*80}")
    
    # 更新args中的confidence阈值
    args.confidence_threshold = confidence_threshold
    
    # 准备输入 - 提取input_ids张量
    input_ids = tokenizer(args.input, return_tensors='pt').input_ids
    
    # 记录开始时间
    start_time = time.time()
    
    # 生成文本
    try:
        generated_text, stats = generate(uav_node, stub, input_ids, tokenizer, args, return_stats=True)
        success = True
        
        # 从stats中提取信息
        total_time = stats['total_time']
        total_tokens = stats['total_tokens']
        throughput = stats['throughput']
        total_rounds = stats['total_rounds']
        avg_gamma = stats['avg_gamma']
        acceptance_rate = stats['acceptance_rate']
        
    except Exception as e:
        print(f"Error during generation: {e}")
        import traceback
        traceback.print_exc()
        generated_text = ""
        success = False
        total_time = 0
        total_tokens = 0
        throughput = 0
        total_rounds = 0
        avg_gamma = 0
        acceptance_rate = 0
    
    # 收集结果
    result = {
        'confidence_threshold': confidence_threshold,
        'success': success,
        'generated_text': generated_text,
        'total_time': total_time,
        'total_tokens': total_tokens,
        'throughput': throughput,
        'total_rounds': total_rounds,
        'avg_gamma': avg_gamma,
        'acceptance_rate': acceptance_rate,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    print(f"\nResults for confidence={confidence_threshold}:")
    print(f"  Throughput: {throughput:.2f} tokens/s")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Total rounds: {total_rounds}")
    print(f"  Avg gamma: {avg_gamma:.2f}")
    print(f"  Acceptance rate: {acceptance_rate:.3f}")
    print(f"  Generated text length: {len(generated_text)} chars")
    
    return result


def run_confidence_sweep(args):
    """运行confidence阈值扫描实验"""
    
    # 设置设备
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # 加载模型
    print(f"\nLoading draft model from {args.draft_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.draft_model_name)
    draft_model = AutoModelForCausalLM.from_pretrained(args.draft_model_name)
    
    # 创建UAV节点
    uav_node = UAVNode(draft_model, device, args)
    
    # confidence阈值列表
    confidence_thresholds = [0.013,0.013]
    
    # 存储所有结果
    all_results = {
        'experiment_name': 'Confidence Threshold Sweep',
        'max_len': args.max_len,
        'input_text': args.input,
        'draft_model': args.draft_model_name,
        'device': args.device,
        'temperature': args.temperature,
        'top_k': args.top_k,
        'top_p': args.top_p,
        'experiments': []
    }
    
    # 连接gRPC服务器
    print(f"\nConnecting to gRPC server at {args.server_addr}...")
    with grpc.insecure_channel(args.server_addr) as channel:
        stub = sd_pb2_grpc.SDVerifyStub(channel)
        
        # 对每个confidence阈值进行测试
        for conf in tqdm(confidence_thresholds, desc="Testing confidence thresholds"):
            result = test_single_confidence(
                uav_node=uav_node,
                stub=stub,
                tokenizer=tokenizer,
                args=args,
                confidence_threshold=conf
            )
            all_results['experiments'].append(result)
            
            # 每次测试后稍微等待一下
            time.sleep(2)
    
    # 打印汇总表
    print_summary_table(all_results)
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_file = f"confidence_sweep_results_{timestamp}.json"
    csv_file = f"confidence_sweep_results_{timestamp}.csv"
    
    save_results_to_json(all_results, json_file)
    save_results_to_csv(all_results, csv_file)
    
    # 找出最佳confidence
    best_result = max(all_results['experiments'], key=lambda x: x['throughput'])
    print(f"\n{'='*80}")
    print(f"BEST PERFORMANCE")
    print(f"{'='*80}")
    print(f"Confidence threshold: {best_result['confidence_threshold']}")
    print(f"Throughput: {best_result['throughput']:.2f} tokens/s")
    print(f"{'='*80}\n")
    
    return all_results


def parse_arguments():
    parser = argparse.ArgumentParser(description='Confidence Threshold Sweep Test')
    
    # 模型参数
    parser.add_argument('--input', type=str, 
                        default="Alan Turing theorized that computers would one day become ")
    parser.add_argument('--draft_model_name', type=str, default="./LLM/opt-125m")
    parser.add_argument('--max_len', type=int, default=1000,
                        help='Maximum length to generate (default: 1000)')
    
    # 生成参数
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--top_p', type=float, default=0)
    parser.add_argument('--seed', type=int, default=321)
    
    # 设备和gRPC
    parser.add_argument('--device', type=str, default='mps')
    parser.add_argument('--server_addr', type=str, required=True,
                        help='gRPC server address, e.g., 127.0.0.1:8000')
    
    # 其他
    parser.add_argument('--use_dist_summary', action='store_true')
    parser.add_argument('--no_cache', action='store_true')
    parser.add_argument('--gamma', type=int, default=4,
                        help='Initial gamma (not used in adaptive mode)')
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    
    print("\n" + "="*80)
    print("CONFIDENCE THRESHOLD SWEEP EXPERIMENT")
    print("="*80)
    print(f"Max generation length: {args.max_len}")
    print(f"Draft model: {args.draft_model_name}")
    print(f"Device: {args.device}")
    print(f"Server: {args.server_addr}")
    print(f"Testing confidence thresholds: 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6")
    print("="*80)
    
    # 运行实验
    results = run_confidence_sweep(args)
    
    print("\n✓ Experiment completed!")


if __name__ == "__main__":
    main()
