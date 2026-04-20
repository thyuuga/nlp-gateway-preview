# nlp-gateway/services/message_filter/base.py
"""
消息过滤器基类

使用策略模式支持多语言扩展：
- ChineseFilterStrategy: 中文
- JapaneseFilterStrategy: 日文（未来）
- EnglishFilterStrategy: 英文（未来）
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class FilterResult:
    """过滤结果"""
    should_embed: bool
    score: int
    coherence: float
    sentence_type: Optional[str] = None  # SVO, SVP, EXIST, etc.
    elements: List[str] = field(default_factory=list)  # time, location, person
    features: List[str] = field(default_factory=list)  # 加分项
    penalties: List[str] = field(default_factory=list)  # 减分项
    reason: str = ""

    def to_dict(self):
        return {
            "should_embed": self.should_embed,
            "score": self.score,
            "coherence": self.coherence,
            "sentence_type": self.sentence_type,
            "elements": self.elements,
            "features": self.features,
            "penalties": self.penalties,
            "reason": self.reason,
        }


class FilterStrategy(ABC):
    """过滤策略抽象基类"""

    @property
    @abstractmethod
    def lang(self) -> str:
        """语言代码：zh, ja, en"""
        pass

    @abstractmethod
    def filter(self, text: str, role: str = "user") -> FilterResult:
        """
        过滤消息

        Args:
            text: 消息文本
            role: 'user' | 'assistant'

        Returns:
            FilterResult
        """
        pass

    @abstractmethod
    def check_three_elements(self, text: str) -> List[str]:
        """检测三要素：时间、地点、人物"""
        pass

    @abstractmethod
    def check_sentence_structure(self, text: str) -> tuple:
        """检测句型结构，返回 (句型, 加分)"""
        pass

    @abstractmethod
    def check_coherence(self, text: str) -> float:
        """检测连贯性，返回 0-1 分数"""
        pass


class MessageFilter:
    """消息过滤器主类"""

    _strategies = {}

    def __init__(self, lang: str = "zh"):
        self.lang = lang
        self._strategy = self._get_strategy(lang)

    @classmethod
    def register_strategy(cls, strategy_class):
        """注册策略类"""
        instance = strategy_class()
        cls._strategies[instance.lang] = strategy_class
        return strategy_class

    def _get_strategy(self, lang: str) -> FilterStrategy:
        strategy_class = self._strategies.get(lang)
        if not strategy_class:
            raise ValueError(f"Unsupported language: {lang}. Available: {list(self._strategies.keys())}")
        return strategy_class()

    def filter(self, text: str, role: str = "user") -> FilterResult:
        """过滤消息"""
        return self._strategy.filter(text, role)

    @classmethod
    def available_languages(cls) -> List[str]:
        """返回支持的语言列表"""
        return list(cls._strategies.keys())
