import random
import re
from datasets import load_dataset
from tqdm import tqdm

def load_mini_spider(n_samples=20, seed=42, split="validation"):
    """
    从本地 Spider 数据集中抽 n 条构成 Mini-Spider
    Spider 是 Text-to-SQL 数据集
    """
    path = "/Users/aoliliaoao/Downloads/DSSD-Efficient-Edge-Computing/benchmark/spider"
    
    spider = load_dataset("parquet", data_files={
        "train": f"{path}/train-00000-of-00001.parquet",
        "validation": f"{path}/validation-00000-of-00001.parquet"
    }, split=split)
    
    data = list(spider)
    
    random.seed(seed)
    mini = random.sample(data, min(n_samples, len(data)))
    
    return mini


def format_spider_prompt(item):
    """
    格式化 Spider 数据为 prompt
    Spider 数据格式:
    - question: 自然语言问题
    - query: SQL 查询
    - db_id: 数据库ID
    """
    question = item.get("question", "")
    db_id = item.get("db_id", "")
    
    prompt = f"Convert the following question to SQL query for database '{db_id}':\nQuestion: {question}\nSQL:"
    
    return prompt


def extract_sql_query(text: str):
    """
    从模型输出中提取SQL查询
    """
    text = text.strip()
    
    # 尝试匹配SQL关键字后的内容
    sql_pattern = r"(?:SQL:|Query:)?\s*(SELECT.*?)(?:\n|$)"
    match = re.search(sql_pattern, text, re.IGNORECASE | re.DOTALL)
    
    if match:
        return match.group(1).strip()
    
    # 如果没有找到，返回整个文本
    return text


def normalize_sql(sql: str):
    """
    标准化SQL查询用于比较
    """
    # 转小写
    sql = sql.lower()
    # 移除多余空格
    sql = re.sub(r'\s+', ' ', sql)
    # 移除末尾的分号
    sql = sql.rstrip(';')
    return sql.strip()


def simple_sql_match(pred_sql: str, gold_sql: str):
    """
    简单的SQL匹配（标准化后字符串比较）
    注意：这是一个简化版本，真实评估需要考虑SQL语义等价性
    """
    pred_norm = normalize_sql(pred_sql)
    gold_norm = normalize_sql(gold_sql)
    
    return pred_norm == gold_norm


def calculate_exact_match(pred_sql: str, gold_sql: str):
    """
    计算精确匹配分数
    """
    return 1.0 if simple_sql_match(pred_sql, gold_sql) else 0.0
