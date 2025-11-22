import argparse
from speculative import * 
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import Tuple, List, Dict
import time
import torch.nn.functional as F
from tqdm import tqdm
import csv, time, os, json
from collections import OrderedDict

#1:添加
import grpc
import sd_pb2
import sd_pb2_grpc

#测试
from test.boolq import load_mini_boolq, extract_boolq_answer

def evaluate_on_mini_boolq(uav_node, stub, tokenizer, args, n_samples=20):
    """
    使用 Mini-BoolQ 进行回归测试，验证 gRPC speculative decoding 的正确性。
    """

    dataset = load_mini_boolq(n_samples=n_samples)

    correct = 0

    print("\n=== Running Mini-BoolQ regression test ===\n")

    for item in tqdm(dataset, desc="Mini-BoolQ"):
        passage = item["passage"]
        question = item["question"]
        gt = "true" if item["answer"] else "false"

        prompt = f"{passage}\nQuestion: {question}\nAnswer:"

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids

        # 调用 gRPC speculative decoding
        # output_text = generate(uav_node, stub, input_ids, tokenizer, args)
        output_text = generate0(uav_node, stub, input_ids, tokenizer, args)

        pred = extract_boolq_answer(output_text)

        if pred == gt:
            correct += 1

    print("\n=== Mini-BoolQ Accuracy ===")
    print(f"Accuracy: {correct}/{n_samples} = {correct/n_samples:.3f}")

    return correct / n_samples

class UAVNode:
    """无人机节点 - 负责draft阶段的小模型推理"""
    
    def __init__(self, model, device, args):
        self.model = model.to(device)
        self.device = device
        self.args = args
        
    def draft_step(self, prefix, gamma):
        """
        UAV端执行draft步骤，生成gamma个候选token
        返回：
          x_draft: (1, prefix_len+γ) 
          q_step_logits_stack: (γ, V)
          dup_bytes: 传输字节数
        """
        x = prefix.to(self.device)
        q_stack = []
        
        with torch.no_grad():
            for _ in range(gamma):
                logits = self.model(x).logits  # (1, seq, V)
                q_stack.append(logits[0, -1].cpu())  # 存储最后一行 (V,)
                next_tok = sample(logits[:, -1, :],
                                self.args.temperature, 
                                self.args.top_k, 
                                self.args.top_p)
                x = torch.cat((x, next_tok), dim=1)
        
        q_step_logits = torch.stack(q_stack, dim=0)  # -> (γ, V)
        
        # 计算传输字节数
        raw_bytes = tensor_nbytes(q_step_logits)
        if self.args.use_dist_summary:
            comp_bytes = sum(tensor_nbytes(compress_logits(row)) for row in q_step_logits)
            dup_bytes = comp_bytes
        else:
            dup_bytes = raw_bytes
            
        return x, q_step_logits, dup_bytes
    
    def draft_DSSD(self, prefix: torch.Tensor, gamma: int) -> Tuple[torch.Tensor, List[float], torch.Tensor]:
        """
        DSSD Draft阶段：生成γ个推测token，记录概率值（上行传输）和完整分布（本地重新采样）
        返回：
          x_draft: 前缀+推测token（shape: (1, seq_len+gamma)）
          q_values: 每个推测token的概率值（list[float], 长度gamma）→ 上行传输
          q_probs: 每个推测token的完整分布（shape: (gamma, V)）→ 本地保存
        """
        torch.manual_seed(111)

        x = prefix.to(self.device)
        q_probs = []  # 保存完整分布（本地用于重新采样）
        q_values = []  # 保存概率值（上行传输）

        with torch.no_grad():
            for i in range(gamma):
                # 1. 小模型前向计算，得到当前分布Q_i(x)
                logits = self.model(x).logits  # (1, seq, V)
                q_dist = F.softmax(logits[0, -1], dim=-1).cpu()  # Q_i(x)：当前步骤的完整分布
                q_probs.append(q_dist)
                
                # 2. 采样下一个token x_i ~ Q_i(x)
                next_tok = sample(logits[:, -1, :], self.args.temperature, self.args.top_k, self.args.top_p)
                x = torch.cat((x, next_tok), dim=1)
                
                # 3. 记录x_i的概率值q_i = Q_i(x_i)（用于上行传输）
                tok_id = next_tok.item()
                q_val = q_dist[tok_id].item()
                q_values.append(q_val)  # q_i = P_Q(x_i)

        # 转换为张量（便于后续处理）
        q_probs = torch.stack(q_probs, dim=0)  # (gamma, V)

        return x, q_values, q_probs

    def resample_DSSD(self, j: int, pj: torch.Tensor, q_probs: torch.Tensor) -> torch.Tensor:
        """
        DSSD Resample阶段：设备用本地保存的Q_j分布和基站发送的P_j分布重新采样
        参数：
          j: 拒绝位置（1~gamma）
          pj: 基站返回的P_j分布（shape: (V,)）
          q_probs: 设备保存的Q分布（shape: (gamma, V)）
        返回：
          xj_prime: 重新采样的token（shape: (1, 1)）
        """

        torch.manual_seed(444)

        q_j = q_probs[j-1]  # Q_j分布（对应拒绝位置j）
        diff = (pj - q_j).clamp(min=0)  # diff = max(0, P_j - Q_j)
        diff = diff / diff.sum()  # 归一化
        xj_prime = torch.multinomial(diff, 1).unsqueeze(0)  # 采样x_j'

        return xj_prime

def generate0(uav_node: UAVNode, stub: sd_pb2_grpc.SDVerifyStub, input_ids: torch.Tensor, tokenizer: AutoTokenizer, args: argparse.Namespace) -> None:

    input_ids = input_ids.to(uav_node.device)
    max_total_len = args.max_len + input_ids.shape[1]  # 生成的总长度（输入+输出）


    # 初始化统计指标
    total_comm_delay = 0.0  # 总通信延迟（上行+下行）
    total_slm_time = 0.0    # 设备小模型（SLM）总时间
    total_llm_time = 0.0    # 基站大模型（LLM）总时间
    rounds = 0              # 循环轮次
    correct_num_total = 0   # 接受的token总数
    reject_num_total = 0    # 拒绝的token总数

    # 初始化前缀
    prefix = input_ids

    with tqdm(total=max_total_len, desc="SD: Speculative Decoding") as pbar:
        pbar.update(prefix.shape[1])  # 初始进度（输入长度）
        initial_len = prefix.shape[1]
        start_time = time.time()

        # 主循环：直到生成达到最大长度
        while prefix.shape[1] < max_total_len:
            old_len = prefix.shape[1]
            rounds += 1

            # print(f"\n{'='*60}")
            # print(f"Round {rounds}")
            # print(f"{'='*60}")

            # ========== 步骤1: 逐个生成gamma个token ==========
            # 打印prefix的文本内容
            # prefix_text = tokenizer.decode(prefix[0], skip_special_tokens=True)
            # print(f"[初始状态] prefix: \"{prefix_text}\"")

            t_slm_start = time.time()
            x_draft, q_values, q_probs = uav_node.draft_DSSD(prefix, args.gamma)
            total_slm_time += time.time() - t_slm_start

            # ========== 步骤2: 准备验证请求 ==========
            tokens_to_send = x_draft.squeeze(0).tolist()

            req = sd_pb2.VerifyRequest(
                x_draft=tokens_to_send,  
                gamma=int(args.gamma),
                q_values=q_values
            )
            

            # ========== 步骤3: 同步发送验证请求 ==========
            comm_start = time.time()
            
            try:
                resp = stub.Verify(req)  # 同步调用
                comm_end = time.time()
            except grpc.RpcError as e:
                print(f"gRPC error: {e}")
                break
            http_dur = (comm_end - comm_start)
            llm_time = float(resp.llm_time)
            total_llm_time += llm_time
            comm_only = max(http_dur - llm_time, 0.0)
            total_comm_delay += comm_only

            flag = int(resp.flag)
            correct_num_total += int(resp.correct_num)
            reject_num_total += int(resp.reject_num)

            # 打印验证结果
            # print(f"\n[步骤4] 验证结果: flag={flag}, correct={resp.correct_num}, reject={resp.reject_num}")
            # if flag == 1 and len(resp.xj) > 0:
            #     xj_id = int(resp.xj[0])
            #     xj_text = tokenizer.decode([xj_id], skip_special_tokens=True)
            #     print(f"        服务器返回token: \"{xj_text}\" (ID={xj_id})")

            # ========== 步骤6: 处理验证结果 ==========
            prefix_len = prefix.shape[1]  # 当前前缀长度
            if flag == 1:
                # 情况1：全部接受（flag=1）→ 基站返回x_{gamma+1}
                xj_id = int(resp.xj[0])
                xj = torch.tensor([[xj_id]], dtype=torch.long, device=uav_node.device)
                new_prefix = torch.cat([x_draft, xj], dim=1)
                # 检查是否超出限制
                if new_prefix.shape[1] > max_total_len:
                    new_prefix = new_prefix[:, :max_total_len]
            else:
                # 情况2：拒绝（flag=0）→ 基站返回j（int） + P_j分布（tensor）
                # 设备用本地保存的q_probs和基站发送的pj重新采样x_j'
                j = int(resp.j)
                pj = torch.tensor(list(resp.pj), dtype=torch.float32)                 
                xj_prime = uav_node.resample_DSSD(j, pj, q_probs)
                # 更新x_draft中的拒绝token（x_j→x_j'）
                x_draft[:, prefix_len + j - 1] = xj_prime.to(x_draft.device)
                # 新前缀：前缀 + 接受的token + 重新采样的token
                new_prefix = torch.cat([prefix, x_draft[:, prefix_len:prefix_len+j]], dim=1)
            
                #打印
                # xj_prime_text = tokenizer.decode(xj_prime[0], skip_special_tokens=True)
                # print(f"[步骤6] ❌ 部分拒绝，回滚到 j={j}，修正为: \"{xj_prime_text}\"")
            
            # 更新前缀
            prefix = new_prefix

            # #打印最终结果
            # new_prefix_text = tokenizer.decode(new_prefix[0], skip_special_tokens=True)
            # print(f"\n[结果] new_prefix:  \"{new_prefix_text}\"")
            # print(f"       本轮新增tokens: {prefix.shape[1] - old_len}个")

            # 更新进度条
            new_len = prefix.shape[1]
            pbar.update(new_len - old_len)

        # 结果统计
        total_time = time.time() - start_time
        total_tokens = prefix.shape[1] - initial_len  # 生成的token总数（排除输入）
        throughput = total_tokens / total_time if total_time > 0 else 0.0
        acceptance_rate = correct_num_total / (rounds * args.gamma) if (rounds * args.gamma) > 0 else 0.0

    # 解码生成的文本
    generated = tokenizer.decode(prefix[0], skip_special_tokens=True)
    print("\n=== DSSD Results(gRPC) ===")
    print(f"Generated text: \033[91m{generated}\033[0m")
    print(f"Throughput: \033[91m{throughput:.2f}\033[0m tokens/s")
    print(f"Acceptance rate: {acceptance_rate:.3f}")
    print(f"Total rounds: {rounds}")
    # print(f"Total accepted tokens: {correct_num_total}")
    # print(f"Total rejected tokens: {reject_num_total}")
    # print(f"Total proposed tokens: {rounds * args.gamma}")
    # print(f"Accept/Reject ratio: {correct_num_total}/{reject_num_total} = {correct_num_total/max(reject_num_total,1):.2f}")
    print(f"Total communication delay: {total_comm_delay:.2f}s")
    print(f"Total SLM (device) time: {total_slm_time:.2f}s")
    print(f"Total LLM (BS) time: {total_llm_time:.2f}s")
    print(f"Total time: {total_time:.2f}s")
    
    return generated


def generate(uav_node: UAVNode, stub: sd_pb2_grpc.SDVerifyStub, input_ids: torch.Tensor, tokenizer: AutoTokenizer, args: argparse.Namespace) -> None:

    input_ids = input_ids.to(uav_node.device)
    max_total_len = args.max_len + input_ids.shape[1]  # 生成的总长度（输入+输出）

    # 初始化统计指标
    total_comm_delay = 0.0  # 总通信延迟（上行+下行）
    total_slm_time = 0.0    # 设备小模型（SLM）总时间
    total_llm_time = 0.0    # 基站大模型（LLM）总时间
    rounds = 0              # 循环轮次
    correct_num_total = 0   # 接受的token总数
    reject_num_total = 0    # 拒绝的token总数
    parallel_generated_total = 0  # 并行生成的token总数
    parallel_accepted_total = 0   # 并行生成且被接受的token总数

    # 初始化前缀
    prefix = input_ids

    with tqdm(total=max_total_len, desc="SD: Speculative Decoding") as pbar:
        pbar.update(prefix.shape[1])  # 初始进度（输入长度）
        initial_len = prefix.shape[1]
        start_time = time.time()

        q_values = []
        q_probs = []  
        
        # 上一轮并行计算的结果(如果被接受)
        parallel_accepted = False

        old_len = prefix.shape[1]

        # 主循环：直到生成达到最大长度
        while prefix.shape[1] < max_total_len:

            old_len = prefix.shape[1]
            generated_token_num = 0
            rounds += 1
            x_draft = prefix.clone()
            confidence = 1

            if parallel_accepted:
                q_values = q_values_current[-len(q_values_current)+1:] 
                q_probs = q_probs_current[-len(q_probs_current)+1:]
                generated_token_num = len(q_values)
                x_draft = x_draft_current
                for q_val in q_values:
                    confidence *= q_val
                parallel_accepted = False  


            print(f"\n{'='*60}")
            print(f"Round {rounds}")
            print(f"{'='*60}")

            # ========== 步骤1: 生成token ==========
            prefix_text = tokenizer.decode(prefix[0], skip_special_tokens=True)
            print(f"[初始状态] prefix: \"{prefix_text}\"")
            print(f"           prefix长度: {prefix.shape[1]}")
            
            t_slm_start = time.time()
            
            print(f"\n[步骤1] 自适应生成:")

            #根据置信度逐个生成
            while confidence > 0.5:
                x_draft, q_value, q_prob = uav_node.draft_DSSD(x_draft, 1)
                generated_token_num += 1
                q_values.append(q_value[0])  
                q_probs.append(q_prob)
                # 更新置信度
                original_confidence = confidence
                confidence = confidence * q_value[0]
                # 打印token信息
                new_token_id = x_draft[0, -1].item()
                new_token_text = tokenizer.decode([new_token_id], skip_special_tokens=True)
                print(f"        Token {generated_token_num}: \"{new_token_text}\" | q={q_value[0]:.4f} | conf: {original_confidence:.4f}→{confidence:.4f}")
            gamma = generated_token_num
            print(f"        ✓ 共生成 {generated_token_num} 个token，最终置信度={confidence:.4f}")

            total_slm_time += time.time() - t_slm_start


            # ========== 步骤2: 准备验证请求 ==========
            tokens_to_send = x_draft.squeeze(0).tolist()
            stacked_q_probs = torch.stack(q_probs, dim=0)   

            req = sd_pb2.VerifyRequest(
                x_draft=tokens_to_send,  
                gamma=generated_token_num,  
                q_values=q_values
            )
            
            # 打印发送的内容
            print(f"\n[步骤2] 准备验证请求:")
            print(f"        发送的gamma: {gamma}")
            print(f"        发送的q_values长度: {len(q_values)}")
            draft_text = tokenizer.decode(tokens_to_send, skip_special_tokens=True)
            print(f"        发送的x_draft内容: \"{draft_text}\"")

            # ========== 步骤3: 异步发送验证请求 ==========
            comm_start = time.time()
            
            try:
                future = stub.Verify.future(req)
                
                timeout_ms = 10  # 每次检查的超时时间(毫秒)

                # 独立的草稿存储变量
                x_draft_current = x_draft.clone()
                
                parallel_count = 0  # 并行计算次数
                q_values_current = []  # 并行计算的q_values
                q_probs_current = []   # 并行计算的q_probs  
                print(f"\n[步骤3] 异步发送验证请求,开始等待服务器响应...")
                
                while True:
                    try:
                        resp = future.result(timeout=timeout_ms / 1000.0)
                        if parallel_count > 0:
                            print(f"[并行计算] 服务器响应已到达,共执行 {parallel_count} 次并行draft")
                        break
                    except grpc.FutureTimeoutError:
                        parallel_count += 1
                        with torch.no_grad():
                            x_draft_current, q_value, q_prob = uav_node.draft_DSSD(x_draft_current, 1)
                            q_values_current.append(q_value[0])  # q_value是列表,取第一个元素
                            q_probs_current.append(q_prob)       # 保持(1,V)结构,稍后stack成(n,V)
                            # 打印并行生成的token
                            new_token_id = x_draft_current[0, -1].item()
                            new_token_text = tokenizer.decode([new_token_id], skip_special_tokens=True)
                            print(f"            → 并行生成token: \"{new_token_text}\" (ID={new_token_id})")
                    except Exception as e:
                        print(f"[并行计算] 等待验证结果时出错: {e}")
                        break

                comm_end = time.time()
            except grpc.RpcError as e:
                print(f"gRPC error: {e}")
                break

            http_dur = (comm_end - comm_start)
            llm_time = float(resp.llm_time)
            total_llm_time += llm_time
            comm_only = max(http_dur - llm_time, 0.0)
            total_comm_delay += comm_only

            flag = int(resp.flag)
            correct_num_total += int(resp.correct_num)
            reject_num_total += int(resp.reject_num)

            # 打印验证结果
            print(f"\n[步骤4] 验证结果: flag={flag}, correct={resp.correct_num}, reject={resp.reject_num}")
            if flag == 1 and len(resp.xj) > 0:
                xj_id = int(resp.xj[0])
                xj_text = tokenizer.decode([xj_id], skip_special_tokens=True)
                print(f"        服务器返回token: \"{xj_text}\" (ID={xj_id})")

            # ========== 步骤6: 处理验证结果 ==========
            prefix_len = prefix.shape[1]
            draft_len = x_draft.shape[1]
            
            print(f"\n[步骤5] 处理验证结果:")
            
            if flag == 1:
                # 情况1：全部接受(flag=1) → 基站返回x_{gamma+1}
                xj_id = int(resp.xj[0])
                xj = torch.tensor([[xj_id]], dtype=torch.long, device=uav_node.device)
                xj_text = tokenizer.decode([xj_id], skip_special_tokens=True)
                
                # 检查并行计算是否有结果,且服务器返回的token与并行计算的第一个token匹配
                if parallel_count > 0:
                    parallel_first_token_id = x_draft_current[0, draft_len].item()

                    if xj_id == parallel_first_token_id:
                        print(f"[并行优化] ✓ 并行计算结果与服务器匹配!使用并行生成的序列")
                        new_prefix = torch.cat([x_draft, xj], dim=1)
                        parallel_accepted = True
                    else:
                        print(f"[并行优化] ✗ 并行计算结果与服务器不匹配,使用原始方案")
                        new_prefix = torch.cat([x_draft, xj], dim=1)
                        parallel_accepted = False
                else:
                    print(f"        无并行计算,直接使用原始draft + 服务器token")
                    new_prefix = torch.cat([x_draft, xj], dim=1)
                    parallel_accepted = False
                
                # 检查是否超出限制
                if new_prefix.shape[1] > max_total_len:
                    new_prefix = new_prefix[:, :max_total_len]
            else:
                # 情况2：拒绝(flag=0) → 基站返回j(int) + P_j分布(tensor)
                # 设备用本地保存的q_probs和基站发送的pj重新采样x_j'
                j = int(resp.j)
                print(f"        拒绝位置j={j}, 需要重新采样")
                pj = torch.tensor(list(resp.pj), dtype=torch.float32)
                xj_prime = uav_node.resample_DSSD(j, pj, q_probs)
                xj_prime_text = tokenizer.decode([xj_prime.item()], skip_special_tokens=True)
                print(f"        重新采样得到token: \"{xj_prime_text}\" (ID={xj_prime.item()})")
                # 更新x_draft中的拒绝token(x_j→x_j')
                x_draft[:, prefix_len + j - 1] = xj_prime.to(x_draft.device)
                # 新前缀：前缀 + 接受的token + 重新采样的token
                new_prefix = torch.cat([prefix, x_draft[:, prefix_len:prefix_len+j]], dim=1)
                parallel_accepted = False

            # 更新前缀
            prefix = new_prefix

            #打印最终结果
            new_prefix_text = tokenizer.decode(new_prefix[0], skip_special_tokens=True)
            print(f"\n[结果] new_prefix:  \"{new_prefix_text}\"")

            # 更新进度条
            new_len = prefix.shape[1]
            pbar.update(new_len - old_len)

        # 结果统计
        total_time = time.time() - start_time
        total_tokens = prefix.shape[1] - initial_len  # 生成的token总数（排除输入）
        throughput = total_tokens / total_time if total_time > 0 else 0.0
        acceptance_rate = correct_num_total / (rounds * args.gamma) if (rounds * args.gamma) > 0 else 0.0

    # 解码生成的文本
    generated = tokenizer.decode(prefix[0], skip_special_tokens=True)
    
    # 计算并行优化的效率
    parallel_acceptance_rate = parallel_accepted_total / max(parallel_generated_total, 1)
    
    print("\n=== DSSD Results(gRPC) ===")
    print(f"Generated text: \033[91m{generated}\033[0m")
    print(f"Throughput: \033[91m{throughput:.2f}\033[0m tokens/s")
    # print(f"Acceptance rate: {acceptance_rate:.3f}")
    print(f"Total rounds: {rounds}")
    # print(f"Total accepted tokens: {correct_num_total}")
    # print(f"Total rejected tokens: {reject_num_total}")
    # print(f"Total proposed tokens: {rounds * args.gamma}")
    # print(f"Accept/Reject ratio: {correct_num_total}/{reject_num_total} = {correct_num_total/max(reject_num_total,1):.2f}")
    # print(f"\n--- Parallel Computing Stats ---")
    # print(f"Parallel generated tokens: {parallel_generated_total}")
    # print(f"Parallel accepted tokens: {parallel_accepted_total}")
    # print(f"Parallel acceptance rate: {parallel_acceptance_rate:.3f}")
    # print(f"Parallel efficiency: {parallel_accepted_total}/{parallel_generated_total}")
    print(f"\n--- Time Stats ---")
    print(f"Total communication delay: {total_comm_delay:.2f}s")
    print(f"Total SLM (device) time: {total_slm_time:.2f}s")
    print(f"Total LLM (BS) time: {total_llm_time:.2f}s")
    print(f"Total time: {total_time:.2f}s")
    
    return generated


def parse_arguments():
    parser = argparse.ArgumentParser(description='args')

    parser.add_argument('--input', type=str, default="Alan Turing theorized that computers would one day become ")
    parser.add_argument('--draft_model_name', type=str, default="./LLM/opt-125m")
    parser.add_argument('--target_model_name', type=str, default="./LLM/opt-1.3b") 
    parser.add_argument('--max_len', type=int, default=80) 
    parser.add_argument('--verbose', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=321)
    parser.add_argument('--gamma', type=int, default=4)
    parser.add_argument('--rtt', type=float, default=0.02)
    parser.add_argument('--bandwidth', type=float, default=1000, help='Bandwidth in Mbps')
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--top_p', type=float, default=0)
    parser.add_argument('--csv_path', type=str, default="results.csv")
    parser.add_argument('--device', type=str, default='mps')
    parser.add_argument('--use_dist_summary', action='store_true', help='upload compressed distribution instead of raw logits')
    parser.add_argument('--no_cache', action='store_true', help='disable Δ-prompt cache (ablation)')
    #1:添加
    parser.add_argument('--server_addr', type=str, required=True, help='e.g., <A100_IP>:50051')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    torch.cuda.empty_cache()  # 清理未使用的显存

    #1:设备改成 apple 的 meta
    device = torch.device(args.device) 
    
    tokenizer = AutoTokenizer.from_pretrained(args.draft_model_name)
    draft_model = AutoModelForCausalLM.from_pretrained(args.draft_model_name)

    input_ids = tokenizer.encode(args.input, return_tensors='pt')
    
    #1:只需要在meta创建UAV节点
    uav_node = UAVNode(draft_model, device, args)

    #1:gRPC
    with grpc.insecure_channel(args.server_addr) as channel:
        stub = sd_pb2_grpc.SDVerifyStub(channel)
        
        text_parallel = generate(uav_node, stub, input_ids, tokenizer, args)
        # text0 = generate0(uav_node, stub, input_ids, tokenizer, args)

        # print("\n===== Running Mini-BoolQ regression test =====")
        # acc = evaluate_on_mini_boolq(uav_node, stub, tokenizer, args, n_samples=20 )
        # print("Mini-BoolQ test accuracy:", acc)
    