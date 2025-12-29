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
        llm_type: str = "gpt",  # "gpt", "deepseek", "llama", "llama_topic", "llama_mask"
        model_path: str = None,
        api_key: str = None,
        device: str = "cuda",
        max_tokens: int = 300,
        num_retries: int = 3,
        base_url: str = None,  # API base URL (用于DeepSeek等兼容OpenAI的API)
    ):
        """
        Args:
            dataset_name: 数据集名称（cora, citeseer, pubmed等）
            bow_cache_dir: BoW词表缓存目录
            llm_type: LLM类型
            model_path: Llama模型路径（如果使用Llama）
            api_key: OpenAI API密钥（如果使用GPT）
            device: 设备
            max_tokens: 最大生成token数
            num_retries: 生成失败时重试次数
        """
        self.dataset_name = dataset_name
        self.llm_type = llm_type
        self.device = device
        self.max_tokens = max_tokens
        self.num_retries = num_retries

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

        # 加载数据集类别信息
        self.category_names = self._load_category_names()

        # 初始化LLM
        if "llama" in llm_type.lower():
            if model_path is None:
                raise ValueError("model_path must be provided for Llama models")
            self._init_llama(model_path)
        elif "gpt" in llm_type.lower() or "deepseek" in llm_type.lower():
            if api_key is None:
                raise ValueError(f"api_key must be provided for {llm_type} models")
            self.api_key = api_key
            # DeepSeek使用兼容OpenAI的API，但需要指定base_url
            if "deepseek" in llm_type.lower():
                if base_url is None:
                    base_url = "https://api.deepseek.com"  # DeepSeek默认endpoint
                self.client = OpenAI(api_key=api_key, base_url=base_url)
                self.model_name = "deepseek-chat"  # DeepSeek默认模型
                print(f"Using DeepSeek API with base_url: {base_url}")
            else:
                self.client = OpenAI(api_key=api_key)
                self.model_name = "gpt-3.5-turbo"  # GPT默认模型
        else:
            raise ValueError(f"Unsupported llm_type: {llm_type}")

    def _init_llama(self, model_path: str):
        """初始化Llama模型"""
        print(f"Loading Llama model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]
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
        self, used_words: List[str], include_topic: bool = True
    ) -> List[Dict[str, str]]:
        """构建LLM提示"""
        if include_topic and "topic" in self.llm_type.lower():
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
                "Generate a title and an abstract for an academic article.\n"
                + "Ensure the generated content explicitly contains the following words: "
                + ", ".join(f"'{word}'" for word in used_words)
                + ".\n"
                + "These words should appear as specified, without using synonyms, plural forms, or other variants.\n"
                + f"Length limit: {self.max_tokens} words."
                + "\nOutput the TITLE and ABSTRACT without explanation.\nTITLE:...\nABSTRACT:..."
            )

        messages = [
            {
                "role": "system",
                "content": "A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.",
            },
            {"role": "user", "content": user_content},
        ]
        return messages

    def generate_text_gpt(self, messages: List[Dict[str, str]]) -> str:
        """使用GPT/DeepSeek生成文本"""
        response = self.client.chat.completions.create(
            model=getattr(self, "model_name", "gpt-3.5-turbo-1106"),
            messages=messages,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content

    def generate_text_llama(
        self,
        messages: List[Dict[str, str]],
        not_used_words: List[str],
        use_mask: bool = True,
    ) -> str:
        """使用Llama生成文本（支持token屏蔽）"""
        # 构建禁用token列表
        if use_mask and "mask" in self.llm_type.lower():
            # 扩展禁用词（包括大写形式）
            Cap = [word.capitalize() for word in not_used_words]
            not_used_words_ext = not_used_words + Cap

            # 编码为token ID
            not_used_tokens = [
                self.tokenizer.encode(word, add_special_tokens=False)[0]
                for word in not_used_words_ext
                if len(self.tokenizer.encode(word, add_special_tokens=False)) > 0
            ]
            the_not_used_tokens = [
                self.tokenizer.encode(f"the {word}", add_special_tokens=False)[-1]
                for word in not_used_words_ext
                if len(self.tokenizer.encode(f"the {word}", add_special_tokens=False))
                > 0
            ]
            not_used_tokens.extend(the_not_used_tokens)

            custom_processor = RestrictProcessor(self.tokenizer, not_used_tokens)
        else:
            # 不使用屏蔽
            custom_processor = RestrictProcessor(self.tokenizer, [])

        logits_processor = LogitsProcessorList([custom_processor])

        # 编码输入
        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)

        # 生成
        outputs = self.model.generate(
            input_ids,
            max_new_tokens=self.max_tokens,
            eos_token_id=self.terminators,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            logits_processor=logits_processor,
        )

        response = outputs[0][input_ids.shape[-1] :]
        text = self.tokenizer.decode(response, skip_special_tokens=True)
        return text

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
            not_used_words: 禁止使用的词列表
            verbose: 是否打印详细信息

        Returns:
            生成的文本（已清理）
        """
        messages = self.build_prompt(used_words)

        # 第一轮生成
        if "llama" in self.llm_type.lower():
            response = self.generate_text_llama(messages, not_used_words)
        else:
            response = self.generate_text_gpt(messages)

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

        # 迭代修正（最多3轮）
        for retry in range(self.num_retries):
            if len(missing_words) == 0 or use_rate >= 95.0:
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
            if "llama" in self.llm_type.lower():
                response = self.generate_text_llama(messages, not_used_words)
            else:
                response = self.generate_text_gpt(messages)

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
