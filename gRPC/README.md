# 端云协同 DSSD（真实异构 + 真实网络）

- `client.py`：端侧，加载小模型（draft），执行 Draft 和 Resample，通过 HTTP 调用云端校验。
- `server.py`：云侧，加载大模型（target），提供 `/verify` 接口执行 Verify。

- 模型权重：
  - 端侧：`./LLM/opt-125m` 等小模型（放在端机器上）。
  - 云侧：`./LLM/opt-6.7b` / `./LLM/opt-13b` 等大模型（放在云服务器上）。
  - 注意：端侧与云侧需使用兼容的 tokenizer（建议同一模型家族，如 OPT）。


## opt

### 启动云端（A100）

- 生成 gRPC 代码：
  ```bash
  # 安装依赖（与运行脚本的同一 Python）
  python3 -m pip install -r requirements.txt

  # 生成 gRPC 代码（会产生 dssd_pb2.py, dssd_pb2_grpc.py）
  python3 gen_grpc.py
  ```

- 启动 gRPC 服务器：
  ```bash
  python server.py \
    --target_model_name /home/lym/sijia/DSSD-Efficient-Edge-Computing/LLM/opt-6.7b \
    --device cuda:0 \
    --port 50051 

  ```
### 启动端侧（笔记本/边缘设备）

- 运行 gRPC 客户端：
  
  - SSH连接
    ```bash
    ssh -L 8000:127.0.0.1:50051 lym@10.2.91.91
    ssh -L 8000:127.0.0.1:50051 lym@121.237.183.19 
    ```

  - 新建终端，运行脚本
    ```bash
    cd /Users/aoliliaoao/Downloads/DSSD-Efficient-Edge-Computing/gRPC/单卡 && PYTHON_BIN=/Library/Frameworks/Python.framework/Versions/3.9/bin/python3 SERVER_ADDR=127.0.0.1:8000 ./run.sh
    ```


## vicuna68m+vicuna33b

### 启动云端（A100）

- 生成 gRPC 代码：
  ```bash
  # 安装依赖（与运行脚本的同一 Python）
  python3 -m pip install -r requirements.txt

  # 生成 gRPC 代码（会产生 dssd_pb2.py, dssd_pb2_grpc.py）
  python3 gen_grpc.py
  ```

- 启动 gRPC 服务器：
  ```bash
  python server.py \
    --target_model_name /home/lym/sijia/DSSD/DSSD-Efficient-Edge-Computing/LLM/vicuna-33b \
    --device cuda:0 \
    --port 50051 \
    --use_multi_gpu \
    --gpu_ids "0,1,2,3" 
    

  ```
### 启动端侧（笔记本/边缘设备）

- 运行 gRPC 客户端：
  
  - SSH连接
    ```bash
    ssh -L 8000:127.0.0.1:50051 lym@10.2.91.91
    ssh -L 8000:127.0.0.1:50051 lym@121.237.183.19 
    ```

  - 新建终端，运行脚本
    ```bash

    cd /Users/aoliliaoao/Downloads/端云投机编码实验/gRPC && \
    PYTHON_BIN=/Library/Frameworks/Python.framework/Versions/3.9/bin/python3 \
    SERVER_ADDR=127.0.0.1:8000 \
    DRAFT_MODEL=/Users/aoliliaoao/Downloads/DSSD-Efficient-Edge-Computing/LLM/opt-125m \
    ./run_grpc_client_H800.sh 
    ```