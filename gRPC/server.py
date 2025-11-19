import argparse
from gRPC.test.speculative import * 
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
from concurrent import futures

class BSNode:
    """基站节点 - 负责verify阶段的大模型推理"""
    
    def __init__(self, model, device, args):
        self.model = model.to(device) if device is not None else model
        self.device = device
        self.args = args
    
    def verify_DSSD(self, x_draft: torch.Tensor, q_values: List[float], gamma: int) -> Tuple[int, int, torch.Tensor, torch.Tensor, int, int]:
        """
        DSSD Verification阶段：用设备发送的概率值验证推测token（符合Algorithm 2）
        参数：
          x_draft: 设备生成的前缀+推测token（shape: (1, seq_len+gamma)）
          q_values: 设备发送的每个推测token的概率值（list[float], 长度gamma）
          gamma: 推测token数量
        返回：
          j: 处理到的位置（1~gamma+1）
          flag: 是否拒绝（0=拒绝，1=接受）
          pj: 拒绝位置的P_j分布（仅flag=0有效，shape: (V,)）
          xj: 接受时的新采样token（仅flag=1有效，shape: (1,1)）
          correct_num: 接受的token数量
          reject_num: 拒绝的token数量
        """
        correct_num = 0
        reject_num = 0
        prefix_len = x_draft.size(1) - gamma  # 前缀长度 = x_draft长度 - 推测token数

        torch.manual_seed(222)

        # 1. 大模型并行计算P_1到P_{gamma+1}的分布（Algorithm 2的关键优化）
        with torch.no_grad():
            if self.device is not None:
                p_all = self.model(x_draft.to(self.device)).logits.cpu()
            else:
                p_all = self.model(x_draft).logits.cpu()
        p_logits = p_all[0, prefix_len-1 : prefix_len+gamma, :]  # (gamma+1, V) → P_1到P_{gamma+1}的logits
        p_probs = F.softmax(p_logits / self.args.temperature, dim=-1)  # (gamma+1, V) → 转换为概率分布

        # 2. 逐token验证（Algorithm 2的Accept/Reject逻辑）
        flag = 1  # 默认接受
        j = 1
        pj = None
        xj = None

        torch.manual_seed(333)

        for i in range(gamma):
            current_j = i + 1  # 当前验证的位置（1~gamma）
            tok_id = int(x_draft[0, prefix_len + i].item())  # 推测token x_i的ID
            q_i = q_values[i]  # 设备发送的q_i = Q_i(x_i)（关键：仅用概率值）
            p_i = p_probs[i, tok_id].item()  # 基站计算的p_i = P_i(x_i)

            # 接受条件：r_j < min(1, p_i/q_i)（Algorithm 2的核心逻辑）
            if torch.rand(1).item() > min(1.0, p_i / q_i):
                # 拒绝：记录位置j和P_j分布（发送给设备重新采样）
                flag = 0
                j = current_j
                pj = p_probs[i]  # P_j分布（shape: (V,)）
                reject_num += 1
                break
            else:
                # 接受：继续验证下一个token
                correct_num += 1

        # 3. 处理全部接受的情况（j=gamma+1）
        if flag == 1:
            j = gamma + 1  # 处理到gamma+1位置
            xj = torch.multinomial(p_probs[gamma], 1).unsqueeze(0)  # 采样x_{gamma+1}~P_{gamma+1}

        return j, flag, pj, xj, correct_num, reject_num

class SDVerifyService(sd_pb2_grpc.SDVerifyServicer):
    def __init__(self, bs_node: BSNode):
        self.bs_node = bs_node

    def Verify(self, request: sd_pb2.VerifyRequest, context):
        q_values = list(request.q_values) if hasattr(request, 'q_values') else []
        x_draft_list = list(request.x_draft)
        x_draft = torch.tensor(x_draft_list, dtype=torch.long).unsqueeze(0)
        gamma = int(request.gamma)

        t0 = time.time()

        j, flag, pj, xj, correct_num, reject_num = self.bs_node.verify_DSSD(
                x_draft, q_values, gamma
            )
        llm_time = time.time() - t0

        resp = sd_pb2.VerifyResponse(
            j=int(j),
            flag=int(flag),
            llm_time=float(llm_time),
            correct_num=int(correct_num),
            reject_num=int(reject_num),
        )
        if pj is not None:
            resp.pj.extend(pj.tolist())
        if xj is not None:
            resp.xj.extend([int(xj.item())])
        return resp

def serve(args: argparse.Namespace) -> None:
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    
    # 设置随机种子以确保可重现性
    torch.manual_seed(321)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(321)
        torch.cuda.manual_seed_all(321)  # 如果使用多GPU

    if args.use_multi_gpu:
        if args.gpu_ids:
            os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
            print(f"[gRPC Server] Setting CUDA_VISIBLE_DEVICES={args.gpu_ids}")
            torch.cuda.empty_cache()
        
        print(f"[gRPC Server] Loading target model from {args.target_model_name} with device_map='auto' (multi-GPU)...")
        print(f"[gRPC Server] Available GPUs: {torch.cuda.device_count()}")
        
        # 使用 device_map="auto" 自动将模型分配到多张 GPU

        model = AutoModelForCausalLM.from_pretrained(
            args.target_model_name,
            device_map="auto",           # 自动分配到多张 GPU
            # torch_dtype=torch.float16
            # low_cpu_mem_usage=True,    # 减少 CPU 内存使用（可选）
        )
        
        # 打印模型分配情况
        if hasattr(model, 'hf_device_map'):
            print("[gRPC Server] Model device map:")
            for name, device in model.hf_device_map.items():
                print(f"  {name}: {device}")
        
        # 使用多 GPU 时，device 参数被忽略，模型已自动分配
        device_t = None
        bs_node = BSNode(model, device_t, args=args)
    else:
        # 原有的单 GPU 逻辑
        device_t = torch.device(args.device)
        print(f"[gRPC Server] Loading target model from {args.target_model_name} on {device_t}...")
        model = AutoModelForCausalLM.from_pretrained(args.target_model_name)
        bs_node = BSNode(model, device_t, args=args)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=args.workers))
    # 注意：Python gRPC 生成的注册函数名称是下划线格式：add_<ServiceName>Servicer_to_server
    sd_pb2_grpc.add_SDVerifyServicer_to_server(SDVerifyService(bs_node), server)
    bind_addr = f"{args.host}:{args.port}"
    server.add_insecure_port(bind_addr)
    server.start()
    print(f"[gRPC Server] Ready. Listening on {bind_addr}")
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("[gRPC Server] Shutting down...")
        server.stop(0)

def parse_arguments():
    parser = argparse.ArgumentParser(description='args')
    parser.add_argument('--target_model_name', type=str, required=True)
    parser.add_argument('--tokenizer_name', type=str, default=None, help='Tokenizer path')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device for single GPU mode (ignored if --multi_gpu is set)')
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=50051)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--use_multi_gpu', action='store_true', help='Enable multi-GPU model parallelism (requires accelerate)')
    parser.add_argument('--gpu_ids', type=str, default=None, help='Comma-separated GPU IDs to use (e.g., "0,1" for 2 GPUs, "0,1,2,3" for 4 GPUs). Only effective with --multi_gpu')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    serve(args)
