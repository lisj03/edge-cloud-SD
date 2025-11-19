import random
import re
from datasets import load_dataset
from tqdm import tqdm

def load_mini_boolq(n_samples=20, seed=42):
    """
    从本地 BoolQ 数据集中抽 n 条构成 Mini-BoolQ（验证集）
    """
    path = "/home/lym/sijia/DSSD/gRPC/benchmark/boolq"

    boolq = load_dataset(path, split="validation")  # BoolQ 自带 train/validation
    data = list(boolq)

    random.seed(seed)
    mini = random.sample(data, n_samples)

    return mini


def extract_boolq_answer(text: str):
    """
    从模型输出中抽取 true/false，适配 LLM
    """
    text = text.strip().lower()

    if "yes" in text or "true" in text:
        return "true"
    if "no" in text or "false" in text:
        return "false"

    # fallback：找词
    if "true" in text:
        return "true"
    if "false" in text:
        return "false"

    return None

