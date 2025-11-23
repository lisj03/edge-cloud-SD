"""
分析confidence阈值扫描实验结果
"""

import json
import sys
import matplotlib.pyplot as plt
import numpy as np


def load_results(json_file):
    """加载JSON结果文件"""
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_results(results, output_file='confidence_sweep_plot.png'):
    """绘制结果图表"""
    experiments = results['experiments']
    
    # 提取数据
    confidences = [e['confidence_threshold'] for e in experiments]
    throughputs = [e['throughput'] for e in experiments]
    avg_gammas = [e['avg_gamma'] for e in experiments]
    acceptance_rates = [e['acceptance_rate'] for e in experiments]
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Confidence Threshold Sweep Results', fontsize=16, fontweight='bold')
    
    # 1. Throughput vs Confidence
    ax1 = axes[0, 0]
    ax1.plot(confidences, throughputs, 'o-', linewidth=2, markersize=8, color='#2E86AB')
    ax1.set_xlabel('Confidence Threshold', fontsize=12)
    ax1.set_ylabel('Throughput (tokens/s)', fontsize=12)
    ax1.set_title('Throughput vs Confidence Threshold', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    
    # 标注最佳点
    best_idx = throughputs.index(max(throughputs))
    ax1.annotate(f'Best: {throughputs[best_idx]:.2f}',
                 xy=(confidences[best_idx], throughputs[best_idx]),
                 xytext=(10, 10), textcoords='offset points',
                 bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # 2. Average Gamma vs Confidence
    ax2 = axes[0, 1]
    ax2.plot(confidences, avg_gammas, 's-', linewidth=2, markersize=8, color='#A23B72')
    ax2.set_xlabel('Confidence Threshold', fontsize=12)
    ax2.set_ylabel('Average Gamma', fontsize=12)
    ax2.set_title('Average Gamma vs Confidence Threshold', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    
    # 3. Acceptance Rate vs Confidence
    ax3 = axes[1, 0]
    ax3.plot(confidences, acceptance_rates, '^-', linewidth=2, markersize=8, color='#F18F01')
    ax3.set_xlabel('Confidence Threshold', fontsize=12)
    ax3.set_ylabel('Acceptance Rate', fontsize=12)
    ax3.set_title('Acceptance Rate vs Confidence Threshold', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log')
    
    # 4. Summary Table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # 创建表格数据
    table_data = []
    table_data.append(['Conf', 'Throughput', 'Avg γ', 'Accept Rate'])
    for e in experiments:
        table_data.append([
            f"{e['confidence_threshold']:.3f}",
            f"{e['throughput']:.2f}",
            f"{e['avg_gamma']:.2f}",
            f"{e['acceptance_rate']:.3f}"
        ])
    
    table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.2, 0.3, 0.2, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # 设置表头样式
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # 高亮最佳行
    best_row = best_idx + 1
    for i in range(4):
        table[(best_row, i)].set_facecolor('#FFEB3B')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to: {output_file}")
    
    # 可选：显示图表
    # plt.show()


def print_analysis(results):
    """打印详细分析"""
    experiments = results['experiments']
    
    print("\n" + "="*80)
    print("DETAILED ANALYSIS")
    print("="*80)
    
    # 基本统计
    throughputs = [e['throughput'] for e in experiments]
    best_exp = max(experiments, key=lambda x: x['throughput'])
    worst_exp = min(experiments, key=lambda x: x['throughput'])
    
    print(f"\n1. Throughput Statistics:")
    print(f"   Best:  {best_exp['throughput']:.2f} tokens/s (confidence={best_exp['confidence_threshold']})")
    print(f"   Worst: {worst_exp['throughput']:.2f} tokens/s (confidence={worst_exp['confidence_threshold']})")
    print(f"   Average: {np.mean(throughputs):.2f} tokens/s")
    print(f"   Std Dev: {np.std(throughputs):.2f} tokens/s")
    print(f"   Improvement: {(best_exp['throughput']/worst_exp['throughput']-1)*100:.1f}%")
    
    # Gamma分析
    avg_gammas = [e['avg_gamma'] for e in experiments]
    print(f"\n2. Average Gamma Statistics:")
    print(f"   Range: {min(avg_gammas):.2f} - {max(avg_gammas):.2f}")
    print(f"   Average: {np.mean(avg_gammas):.2f}")
    
    # Acceptance Rate分析
    acceptance_rates = [e['acceptance_rate'] for e in experiments]
    print(f"\n3. Acceptance Rate Statistics:")
    print(f"   Range: {min(acceptance_rates):.3f} - {max(acceptance_rates):.3f}")
    print(f"   Average: {np.mean(acceptance_rates):.3f}")
    
    # 相关性分析
    from scipy.stats import pearsonr
    corr_gamma, p_gamma = pearsonr([np.log(e['confidence_threshold']) for e in experiments], avg_gammas)
    corr_accept, p_accept = pearsonr([np.log(e['confidence_threshold']) for e in experiments], acceptance_rates)
    
    print(f"\n4. Correlation Analysis (with log(confidence)):")
    print(f"   Correlation with Avg Gamma: {corr_gamma:.3f} (p={p_gamma:.4f})")
    print(f"   Correlation with Accept Rate: {corr_accept:.3f} (p={p_accept:.4f})")
    
    # 推荐
    print(f"\n5. Recommendation:")
    print(f"   Optimal confidence threshold: {best_exp['confidence_threshold']}")
    print(f"   This provides the best throughput of {best_exp['throughput']:.2f} tokens/s")
    print(f"   with an average gamma of {best_exp['avg_gamma']:.2f}")
    print(f"   and acceptance rate of {best_exp['acceptance_rate']:.3f}")
    
    print("="*80 + "\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_confidence_results.py <results.json>")
        print("\nLooking for latest results file...")
        import glob
        json_files = glob.glob("confidence_sweep_results_*.json")
        if json_files:
            json_file = max(json_files)  # 取最新的
            print(f"Found: {json_file}")
        else:
            print("No results file found!")
            sys.exit(1)
    else:
        json_file = sys.argv[1]
    
    print(f"Loading results from: {json_file}")
    results = load_results(json_file)
    
    # 打印分析
    print_analysis(results)
    
    # 绘制图表
    try:
        plot_results(results)
    except ImportError:
        print("\n⚠️  matplotlib or scipy not installed, skipping plot generation")
        print("   Install with: pip install matplotlib scipy")
    except Exception as e:
        print(f"\n⚠️  Error generating plot: {e}")


if __name__ == "__main__":
    main()
