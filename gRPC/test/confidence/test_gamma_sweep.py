"""
Gamma sweep experiment for speculative decoding.
Iterates over different gamma values, invokes generate0, and compares throughput.
"""

import argparse
import json
import csv
import time
from datetime import datetime

import grpc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import sd_pb2_grpc
from client import UAVNode, generate0


def save_results_to_json(results, output_path: str) -> None:
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Results saved to JSON: {output_path}")


def save_results_to_csv(results, output_path: str) -> None:
    fieldnames = [
        'gamma',
        'throughput',
        'total_time',
        'total_tokens',
        'total_rounds',
        'acceptance_rate',
        'generated_text_length',
        'timestamp'
    ]

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in results['experiments']:
            writer.writerow({
                'gamma': item['gamma'],
                'throughput': item['throughput'],
                'total_time': item['total_time'],
                'total_tokens': item['total_tokens'],
                'total_rounds': item['total_rounds'],
                'acceptance_rate': item['acceptance_rate'],
                'generated_text_length': len(item['generated_text']),
                'timestamp': item['timestamp']
            })

    print(f"✓ Results saved to CSV: {output_path}")


def print_summary_table(results) -> None:
    print("\n" + "=" * 100)
    print("GAMMA SWEEP RESULTS")
    print("=" * 100)
    print(f"{'Gamma':<8} {'Throughput':<15} {'Total Time':<12} {'Tokens':<10} {'Rounds':<10} {'Accept Rate':<12}")
    print("-" * 100)

    for item in results['experiments']:
        print(
            f"{item['gamma']:<8} "
            f"{item['throughput']:<15.2f} "
            f"{item['total_time']:<12.2f} "
            f"{item['total_tokens']:<10} "
            f"{item['total_rounds']:<10} "
            f"{item['acceptance_rate']:<12.3f}"
        )

    print("=" * 100)


def test_single_gamma(uav_node, stub, tokenizer, args, gamma: int):
    print(f"\n{'=' * 80}")
    print(f"Testing gamma = {gamma}")
    print(f"{'=' * 80}")

    args.gamma = gamma
    input_ids = tokenizer(args.input, return_tensors='pt').input_ids

    try:
        generated_text, stats = generate0(
            uav_node=uav_node,
            stub=stub,
            input_ids=input_ids,
            tokenizer=tokenizer,
            args=args,
            return_stats=True
        )
    except Exception as exc:
        print(f"Error during generation for gamma={gamma}: {exc}")
        return {
            'gamma': gamma,
            'success': False,
            'generated_text': '',
            'throughput': 0.0,
            'total_time': 0.0,
            'total_tokens': 0,
            'total_rounds': 0,
            'acceptance_rate': 0.0,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    result = {
        'gamma': gamma,
        'success': True,
        'generated_text': generated_text,
        'throughput': stats['throughput'],
        'total_time': stats['total_time'],
        'total_tokens': stats['total_tokens'],
        'total_rounds': stats['total_rounds'],
        'acceptance_rate': stats['acceptance_rate'],
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    print(
        f"\nResults for gamma={gamma}:\n"
        f"  Throughput: {result['throughput']:.2f} tokens/s\n"
        f"  Total time: {result['total_time']:.2f}s\n"
        f"  Total tokens: {result['total_tokens']}\n"
        f"  Total rounds: {result['total_rounds']}\n"
        f"  Acceptance rate: {result['acceptance_rate']:.3f}\n"
        f"  Generated text length: {len(result['generated_text'])} chars"
    )

    return result


def run_gamma_sweep(args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    print(f"Using device: {device}")
    print(f"Loading draft model from {args.draft_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.draft_model_name)
    draft_model = AutoModelForCausalLM.from_pretrained(args.draft_model_name)
    uav_node = UAVNode(draft_model, device, args)

    all_results = {
        'experiment_name': 'Gamma Sweep',
        'input_text': args.input,
        'max_len': args.max_len,
        'draft_model': args.draft_model_name,
        'device': args.device,
        'temperature': args.temperature,
        'top_k': args.top_k,
        'top_p': args.top_p,
        'gamma_values': args.gamma_values,
        'experiments': []
    }

    print(f"\nConnecting to gRPC server at {args.server_addr}...")
    with grpc.insecure_channel(args.server_addr) as channel:
        stub = sd_pb2_grpc.SDVerifyStub(channel)

        for gamma in args.gamma_values:
            result = test_single_gamma(uav_node, stub, tokenizer, args, gamma)
            all_results['experiments'].append(result)
            if args.cooldown > 0:
                time.sleep(args.cooldown)

    print_summary_table(all_results)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = args.save_json or f"gamma_sweep_results_{timestamp}.json"
    csv_path = args.save_csv or f"gamma_sweep_results_{timestamp}.csv"

    save_results_to_json(all_results, json_path)
    save_results_to_csv(all_results, csv_path)

    best = max(all_results['experiments'], key=lambda item: item['throughput']) if all_results['experiments'] else None
    if best:
        print(f"\n{'=' * 80}")
        print("BEST THROUGHPUT")
        print(f"Gamma: {best['gamma']} | Throughput: {best['throughput']:.2f} tokens/s")
        print(f"{'=' * 80}")

    return all_results


def parse_arguments():
    parser = argparse.ArgumentParser(description='Gamma sweep for speculative decoding (generate0).')

    parser.add_argument('--input', type=str,
                        default="Alan Turing theorized that computers would one day become ")
    parser.add_argument('--draft_model_name', type=str, default="./LLM/opt-125m")
    parser.add_argument('--max_len', type=int, default=200)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--top_p', type=float, default=0)
    parser.add_argument('--seed', type=int, default=321)
    parser.add_argument('--gamma', type=int, default=4,
                        help='baseline gamma, will be overridden during sweep')
    parser.add_argument('--gamma_values', type=int, nargs='+', default=[2, 4, 6, 8, 10])
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--server_addr', type=str, required=True)
    parser.add_argument('--use_dist_summary', action='store_true')
    parser.add_argument('--no_cache', action='store_true')
    parser.add_argument('--cooldown', type=float, default=1.0,
                        help='seconds to sleep between runs')
    parser.add_argument('--save_json', type=str, default=None,
                        help='optional path to save JSON results')
    parser.add_argument('--save_csv', type=str, default=None,
                        help='optional path to save CSV results')

    return parser.parse_args()


def main():
    args = parse_arguments()
    run_gamma_sweep(args)


if __name__ == '__main__':
    main()
