"""
Text Attack Generator: 将 WTGIA 的文本生成方法集成到 heir attack
用于从梯度扰动生成对抗性文本属性（非注入攻击）
"""

import os
import re
import torch
import torch.nn.functional as F
import numpy as np
import pickle
from tqdm import tqdm
from typing import List, Tuple, Optional, Dict
from sklearn.feature_extraction.text import CountVectorizer
from openai import OpenAI
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LogitsProcessorList,
    LogitsProcessor,
)


class RestrictProcessor(LogitsProcessor):
    """禁止特定token的Logits处理器"""

    def __init__(self, tokenizer, non_target_tokens):
        self.tokenizer = tokenizer
        self.non_target_tokens = non_target_tokens

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        scores[:, self.non_target_tokens] = -float("Inf")
        return scores


class TextAttackGenerator:
    """文本攻击生成器：从属性扰动生成对抗性文本"""

    def __init__(
        self,
        dataset_name: str,
        bow_cache_dir: str = "./bow_cache",
        api_key: str = None,
        base_url: str = None,
        device: str = "cuda",
        max_tokens: int = 50,
        num_retries: int = 0,
        llm_type: str = "gpt",
        model_path: str = None,
        feature_dim: int = None,
    ):
        """
        Args:
            dataset_name: 数据集名称（cora, citeseer, pubmed等）
            bow_cache_dir: BoW词表缓存目录
            api_key: OpenAI兼容API密钥
            base_url: API基础URL（如 https://api.openai.com/v1）
            device: 设备
            max_tokens: 最大生成token数
            num_retries: 生成失败时重试次数
            llm_type: LLM类型（"gpt" 或 "llama"）
            model_path: 本地模型路径（llm_type="llama"时需要）
            feature_dim: 数据集特征维度（用于对齐BoW向量）
        """
        self.dataset_name = dataset_name
        self.device = device
        self.max_tokens = max_tokens
        self.num_retries = num_retries
        self.llm_type = llm_type.lower()
        self.feature_dim = feature_dim

        # 加载BoW词表
        vectorizer_path = os.path.join(bow_cache_dir, f"{dataset_name}.pkl")
        if os.path.exists(vectorizer_path):
            with open(vectorizer_path, "rb") as f:
                self.vectorizer = pickle.load(f)
            self.vocab = self.vectorizer.get_feature_names_out()
            print(f"Loaded BoW vocabulary: {len(self.vocab)} words")
        else:
            raise FileNotFoundError(
                f"BoW vocabulary not found at {vectorizer_path}. "
                f"Please run data preprocessing first."
            )

        # 记录词表大小
        self.vocab_size = len(self.vocab)
        print(f"Vocab size: {self.vocab_size}, Feature dim: {self.feature_dim}")

        # 加载数据集类别信息
        self.category_names = self._load_category_names()

        # 初始化LLM
        if self.llm_type == "gpt":
            if api_key is None:
                raise ValueError("api_key must be provided for GPT")
            self.client = OpenAI(api_key=api_key, base_url=base_url)

            # 检测是否为Ollama（通过base_url判断）
            if base_url and "localhost:11434" in base_url:
                # Ollama模式：从环境变量或参数获取模型名
                self.model_name = os.environ.get(
                    "OLLAMA_MODEL", "llama3.2:1b-instruct-fp16"
                )
                print(f"🦙 Detected Ollama - using model: {self.model_name}")
            else:
                # 标准GPT
                self.model_name = "gpt-3.5-turbo"

            self.model = None
            self.tokenizer = None
            print(f"Initialized GPT client with model: {self.model_name}")
        elif self.llm_type == "llama":
            if model_path is None:
                raise ValueError("model_path must be provided for Llama")
            self._init_llama(model_path)
            self.client = None
            print(f"Initialized Llama model from: {model_path}")
        else:
            raise ValueError(f"Unsupported llm_type: {llm_type}. Use 'gpt' or 'llama'")

    def _init_llama(self, model_path: str):
        """初始化Llama模型"""
        print(f"Loading Llama model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            # attn_implementation="flash_attention_2"  # 如果需要更快的推理
        )
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]
        self.model.eval()  # 设置为评估模式
        print("Llama model loaded successfully")

    def _load_category_names(self) -> List[str]:
        """加载数据集的类别名称"""
        mappings = {
            "cora": [
                "Rule Learning",
                "Neural Networks",
                "Case Based",
                "Genetic Algorithms",
                "Theory",
                "Reinforcement Learning",
                "Probabilistic Methods",
            ],
            "citeseer": [
                "Agents",
                "Machine Learning",
                "Information Retrieval",
                "Database",
                "Human Computer Interaction",
                "Artificial Intelligence",
            ],
            "pubmed": [
                "Diabetes Mellitus, Experimental",
                "Diabetes Mellitus Type 1",
                "Diabetes Mellitus Type 2",
            ],
        }
        return mappings.get(self.dataset_name.lower(), ["Category"])

    def extract_words_from_bow_vector(
        self, bow_vector: np.ndarray
    ) -> Tuple[List[str], List[str]]:
        """
        从BoW向量中提取必须使用和禁止使用的词

        Args:
            bow_vector: [vocab_size] BoW向量（0/1或连续值）

        Returns:
            used_words: 必须使用的词列表（bow_vector > threshold）
            not_used_words: 禁止使用的词列表（bow_vector == 0或很小）
        """
        if isinstance(bow_vector, torch.Tensor):
            bow_vector = bow_vector.detach().cpu().numpy()

        # 判断是否为二值向量
        is_binary = np.all((bow_vector == 0) | (bow_vector == 1))

        if is_binary:
            used_words = [
                self.vocab[i] for i in range(len(self.vocab)) if bow_vector[i] == 1
            ]
            not_used_words = [
                self.vocab[i] for i in range(len(self.vocab)) if bow_vector[i] == 0
            ]
        else:
            # 连续值：使用阈值
            threshold = 0.1
            used_words = [
                self.vocab[i]
                for i in range(len(self.vocab))
                if bow_vector[i] > threshold
            ]
            not_used_words = [
                self.vocab[i]
                for i in range(len(self.vocab))
                if bow_vector[i] <= threshold
            ]

        return used_words, not_used_words

    def extract_words_from_gradient(
        self,
        original_bow: np.ndarray,
        gradient: np.ndarray,
        top_k: int = 20,
        use_gradient_for_selection: bool = True,
    ) -> Tuple[List[str], List[str]]:
        """
        从梯度中提取关键词（heir attack特定方法）

        Args:
            original_bow: [vocab_size] 原始BoW向量
            gradient: [vocab_size] 梯度向量
            top_k: 选择top-k个词
            use_gradient_for_selection: 是否使用梯度来选择词（True=基于梯度，False=基于原始值）

        Returns:
            must_use_words: 必须使用的词（梯度最大/最希望变为1的词）
            must_not_use_words: 禁止使用的词（梯度最小/最希望变为0的词）
        """
        if isinstance(original_bow, torch.Tensor):
            original_bow = original_bow.detach().cpu().numpy()
        if isinstance(gradient, torch.Tensor):
            gradient = gradient.detach().cpu().numpy()

        if use_gradient_for_selection:
            # 基于梯度选择：
            # - 对于0->1：梯度大的词（模型希望增加）
            # - 对于1->0：梯度小的词（模型希望减少）

            # 选择希望从0->1的词（当前为0，梯度正向大）
            zero_mask = original_bow == 0
            pos_scores = np.where(zero_mask, gradient, -np.inf)
            top_pos_idx = np.argsort(pos_scores)[-top_k:]
            must_use_words = [
                self.vocab[i] for i in top_pos_idx if pos_scores[i] > -np.inf
            ]

            # 选择希望从1->0的词（当前为1，梯度负向大）
            one_mask = original_bow == 1
            neg_scores = np.where(one_mask, -gradient, -np.inf)
            top_neg_idx = np.argsort(neg_scores)[-top_k:]
            must_not_use_words = [
                self.vocab[i] for i in top_neg_idx if neg_scores[i] > -np.inf
            ]

        else:
            # 基于原始值选择（简单方法）
            must_use_words = [
                self.vocab[i] for i in range(len(self.vocab)) if original_bow[i] == 1
            ][:top_k]
            must_not_use_words = [
                self.vocab[i] for i in range(len(self.vocab)) if original_bow[i] == 0
            ][:top_k]

        return must_use_words, must_not_use_words

    def build_prompt(
        self, used_words: List[str], include_topic: bool = False
    ) -> List[Dict[str, str]]:
        """构建LLM提示"""
        if include_topic:
            user_content = (
                f"There are {len(self.category_names)} types of paper, which are "
                + ", ".join(self.category_names)
                + ".\n"
                + "Generate a title and an abstract for paper belongs to one of the given categories.\n"
                + "Ensure the generated content explicitly contains the following words: "
                + ", ".join(f"'{word}'" for word in used_words)
                + ".\n"
                + "These words should appear as specified, without using synonyms, plural forms, or other variants.\n"
                + f"Length limit: {self.max_tokens} words."
                + "\nOutput the TITLE and ABSTRACT without explanation.\nTITLE:...\nABSTRACT:..."
            )
        else:
            user_content = (
                "Write a short academic text using these words: "
                + ", ".join(used_words[:15])  # 最多15个词
                + ".\nOutput text only, no explanation."
            )

        messages = [
            {
                "role": "system",
                "content": "A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.",
            },
            {"role": "user", "content": user_content},
        ]
        return messages

    def generate_cluster_template(
        self,
        cluster_attributes: List[str],
        discriminative_words: List[str],
        style_constraints: str = "Keep it concise, academic, and natural.",
        num_candidates: int = 3,
    ) -> List[str]:
        """
        Generate cluster-level templates.

        Args:
            cluster_attributes: List of important attributes for the cluster
            discriminative_words: List of discriminative words for the cluster
            style_constraints: Style constraints string
            num_candidates: Number of templates to generate

        Returns:
            List of generated templates
        """
        prompt = (
            f"Task: Generate {num_candidates} distinct text templates for a cluster of academic papers.\n"
            f"Cluster Keywords: {', '.join(cluster_attributes[:10])}\n"
            f"Discriminative Words (Must Include): {', '.join(discriminative_words[:10])}\n"
            f"Style: {style_constraints}\n\n"
            f"Requirements:\n"
            f"1. Write {num_candidates} different templates. Each template should be a coherent paragraph (Abstract-like).\n"
            f"2. Integrate the 'Discriminative Words' naturally.\n"
            f"3. You can use placeholders like [DETAILS] for specific node information, but the text should be mostly complete.\n"
            f"4. Output ONLY the templates, separated by '|||'. Do not add numbering or labels like 'Template 1'.\n"
            f"5. Do not include explanations."
        )

        messages = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant for text generation.",
            },
            {"role": "user", "content": prompt},
        ]

        try:
            response_text = self.generate_text(messages)
            # Parse response
            templates = [t.strip() for t in response_text.split("|||") if t.strip()]
            # Fallback if separator not found but text exists
            if not templates and response_text:
                templates = [response_text.strip()]
            return templates
        except Exception as e:
            print(f"Error generating cluster templates: {e}")
            return []

    def generate_text(self, messages: List[Dict[str, str]]) -> str:
        """调用LLM生成文本"""
        if self.llm_type == "gpt":
            # 使用GPT API
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content
        elif self.llm_type == "llama":
            # 使用本地Llama模型
            input_ids = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            ).to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    max_new_tokens=self.max_tokens,
                    eos_token_id=self.terminators,
                    pad_token_id=self.tokenizer.pad_token_id,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.9,
                )

            response = outputs[0][input_ids.shape[-1] :]
            text = self.tokenizer.decode(response, skip_special_tokens=True)
            return text
        else:
            raise ValueError(f"Unsupported llm_type: {self.llm_type}")

    def calculate_usage_rates(
        self, text: str, should_use_words: List[str], should_not_use_words: List[str]
    ) -> Tuple[float, float, List[str]]:
        """计算词的使用率"""
        text_lower = text.lower().split()
        text_words = [subpart for part in text_lower for subpart in part.split("-")]

        should_use_count = 0
        missing_words = []
        for word in should_use_words:
            if word.lower() in text_words:
                should_use_count += 1
            else:
                missing_words.append(word)

        should_not_use_count = 0
        for word in should_not_use_words:
            if word.lower() in text_words:
                should_not_use_count += 1

        should_use_rate = (
            (should_use_count / len(should_use_words)) * 100
            if len(should_use_words) > 0
            else 100.0
        )
        should_not_use_rate = (
            (should_not_use_count / len(should_not_use_words)) * 100
            if len(should_not_use_words) > 0
            else 0.0
        )

        return should_use_rate, should_not_use_rate, missing_words

    def clear_text(self, raw_text: str) -> str:
        """清理生成的文本"""
        raw_text = re.sub(r"\btitle:\s*", "", raw_text, flags=re.IGNORECASE)
        raw_text = re.sub(r"\btitle\b", "", raw_text, flags=re.IGNORECASE)
        raw_text = re.sub(r"\babstract:\s*", "", raw_text, flags=re.IGNORECASE)
        raw_text = re.sub(r"\babstract\b", "", raw_text, flags=re.IGNORECASE)
        raw_text = raw_text.replace("\n", " ")
        raw_text = raw_text.replace('"', "")
        return raw_text.strip()

    def generate_adversarial_text(
        self,
        used_words: List[str],
        not_used_words: List[str],
        verbose: bool = False,
    ) -> str:
        """
        生成对抗性文本（带反馈修正）

        Args:
            used_words: 必须使用的词列表
            not_used_words: 禁止使用的词列表（当前实现中不强制，仅作为参考）
            verbose: 是否打印详细信息

        Returns:
            生成的文本（已清理）
        """
        messages = self.build_prompt(used_words)

        # 第一轮生成
        response = self.generate_text(messages)

        use_rate, not_use_rate, missing_words = self.calculate_usage_rates(
            response, used_words, not_used_words
        )

        if verbose:
            print(
                f"Initial - Use Rate: {use_rate:.2f}%, Not Use Rate: {not_use_rate:.2f}%"
            )

        messages.append({"role": "assistant", "content": response})
        best_response = response
        best_use_rate = use_rate

        # 迭代修正（仅在效果很差时重试）
        for retry in range(self.num_retries):
            if len(missing_words) == 0 or use_rate >= 80.0:
                break

            # 构建反馈
            feedback = (
                f"You forgot to use "
                + ", ".join(f"'{word}'" for word in missing_words)
                + ".\n"
                + "Output the corrected TITLE and ABSTRACT without explanation.\nTITLE:...\nABSTRACT:..."
            )
            messages.append({"role": "user", "content": feedback})

            # 重新生成
            response = self.generate_text(messages)

            use_rate, not_use_rate, missing_words = self.calculate_usage_rates(
                response, used_words, not_used_words
            )

            if verbose:
                print(
                    f"Retry {retry+1} - Use Rate: {use_rate:.2f}%, Not Use Rate: {not_use_rate:.2f}%"
                )

            messages.append({"role": "assistant", "content": response})

            if use_rate > best_use_rate:
                best_use_rate = use_rate
                best_response = response

        # 清理并返回
        return self.clear_text(best_response)

    def batch_generate_adversarial_texts(
        self,
        bow_vectors: np.ndarray,
        gradients: Optional[np.ndarray] = None,
        top_k: int = 20,
        use_gradient: bool = True,
        verbose: bool = False,
    ) -> List[str]:
        """
        批量生成对抗性文本

        Args:
            bow_vectors: [n, vocab_size] BoW向量
            gradients: [n, vocab_size] 梯度向量（可选）
            top_k: 选择top-k个词
            use_gradient: 是否使用梯度选择词
            verbose: 是否打印进度

        Returns:
            生成的文本列表
        """
        n = bow_vectors.shape[0]
        generated_texts = []

        iterator = (
            tqdm(range(n), desc="Generating adversarial texts") if verbose else range(n)
        )

        for i in iterator:
            if use_gradient and gradients is not None:
                # 使用梯度选择词（heir attack方法）
                used_words, not_used_words = self.extract_words_from_gradient(
                    bow_vectors[i],
                    gradients[i],
                    top_k=top_k,
                    use_gradient_for_selection=True,
                )
            else:
                # 直接从BoW向量提取（WTGIA原始方法）
                used_words, not_used_words = self.extract_words_from_bow_vector(
                    bow_vectors[i]
                )
                # 限制词数
                used_words = used_words[:top_k]
                not_used_words = not_used_words[:top_k]

            # 生成文本
            try:
                text = self.generate_adversarial_text(
                    used_words, not_used_words, verbose=False
                )
                generated_texts.append(text)
            except Exception as e:
                print(f"Error generating text for node {i}: {e}")
                # 失败时返回空字符串或原始文本
                generated_texts.append("")

        return generated_texts
