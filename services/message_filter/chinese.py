# nlp-gateway/services/message_filter/chinese.py
"""
中文消息过滤策略（基于 HanLP SRL + POS）

P0 改进版 v2：
- 零宽字符/空白 normalize
- assistant 更硬门禁（直接拒绝，除非有高价值特征）
- RELATIONSHIP_KEYWORDS 在 strip 前后都判断
- POS 密度计算排除功能词
- 长度改用 token 数
- fallback 标记 features
"""

import re
import logging
from typing import List, Set
from .base import FilterStrategy, FilterResult, MessageFilter

logger = logging.getLogger("nlp_gateway.message_filter")

# 延迟导入 HanLP
_hanlp_pipeline = None


def get_hanlp():
    """延迟加载 HanLP pipeline"""
    global _hanlp_pipeline
    if _hanlp_pipeline is None:
        import hanlp
        _hanlp_pipeline = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH)
    return _hanlp_pipeline


# ========== 预处理 ==========

def strip_actions(text: str) -> str:
    """去掉括号内的动作描写"""
    return re.sub(r'[（(][^）)]*[）)]', '', text).strip()


def normalize_text(text: str) -> str:
    """
    文本标准化：
    - 去除零宽字符
    - 全角空格转半角
    - 连续空白压缩
    """
    # 去除零宽字符
    text = text.replace('\u200b', '').replace('\ufeff', '').replace('\u200c', '').replace('\u200d', '')
    # 全角空格转半角
    text = text.replace('\u3000', ' ')
    # 连续空白压缩
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ========== 快速过滤模式 ==========

# 纯语气词/应答词
LOW_VALUE_PATTERNS = [
    r'^[嗯哦啊呀哈嘿呃唔嘛吧啦呢吗噢哇欸]+[。！？~…、，,.!?]*$',
    r'^(好|行|是|对|没|不|可以|知道了?|明白了?|好的|行的|是的|对的|嗯嗯|哦哦|好吧|行吧)[。！？~…]*$',
    r'^(谢谢|感谢|抱歉|对不起|不好意思)[。！？~…]*$',
    r'^[.。！!？?~…]+$',
]

# 存在句/在不在
EXISTENCE_PATTERNS = [
    r'^(我在|在呢|我在的|在的|我在哦|我在呀|在吗|我在这|我就在|我一直在|还在|都在)[。！？~…哦呢呀]*$',
    r'^(收到|了解|明白|懂了|OK|ok|好嘞|好哒|知道啦)[。！？~…]*$',
]

# 称呼/唤醒词
GREETING_PATTERNS = [
    r'^(喂|在吗|嗨|hi|hello|你好)[。！？~…]*$',
    r'^[？?]+$',
]

# 关系宣言关键词（在 strip 前后都判断）
RELATIONSHIP_KEYWORDS = ['男朋友', '女朋友', '老婆', '老公', '爱你', '喜欢你', '抱抱', '亲亲', 'mua', '么么', '爱死你', '想你']

# 编译所有低价值模式
ALL_LOW_VALUE_PATTERNS = LOW_VALUE_PATTERNS + EXISTENCE_PATTERNS + GREETING_PATTERNS
LOW_VALUE_PATTERN = re.compile('|'.join(ALL_LOW_VALUE_PATTERNS), re.IGNORECASE)

# 纯 emoji 模式（避免浪费 HanLP 资源）
EMOJI_ONLY_PATTERN = re.compile(r'^[\U00010000-\U0010ffff\s\u200d\ufe0f]+$')

# 高价值快速特征（用于 fallback 白名单 + assistant 白名单）
HIGH_VALUE_FAST_PATTERNS = [
    r'\d',  # 含数字
    r'(今天|明天|昨天|后天|上午|下午|晚上|几点|点钟|\d+月|\d+号|\d+日|周[一二三四五六日]|星期)',  # 时间词
    r'(公司|学校|医院|机场|车站|健身房|餐厅|咖啡|超市|商场|家里|东京|大阪|北京|上海|福冈|济南)',  # 地点词
    r'https?://|www\.|@.*\.',  # URL/邮箱
]
HIGH_VALUE_FAST_PATTERN = re.compile('|'.join(HIGH_VALUE_FAST_PATTERNS))

# 事实锚点词（关系宣言句需要有这些才通过）
FACT_ANCHOR_PATTERN = re.compile(r'(\d|今天|明天|昨天|第一次|刚才|之前|以前|那时|那天|公司|学校|健身房)')

# POS 内容词标签（CTB 标注体系）
CONTENT_POS_TAGS = {'NN', 'NR', 'NT', 'VV', 'VA', 'AD', 'JJ', 'CD', 'OD'}

# POS 功能词标签（计算密度时排除）
FUNCTION_POS_TAGS = {'DEC', 'DEG', 'DEV', 'AS', 'SP', 'PU', 'CC', 'CS', 'ETC', 'IJ', 'ON', 'LB', 'SB', 'BA', 'MSP'}


@MessageFilter.register_strategy
class ChineseFilterStrategy(FilterStrategy):
    """中文过滤策略（P0 改进版 v2）"""

    @property
    def lang(self) -> str:
        return "zh"

    def filter(self, text: str, role: str = "user") -> FilterResult:
        """
        主过滤逻辑

        角色门禁：
        - assistant: 直接拒绝，除非命中 HIGH_VALUE_FAST_PATTERN
        - user: 阈值 3

        评分：
        - SRL 完整（ARG0 + 谓词 + ARG1）: +3
        - SRL 部分（谓词 + ARG1）: +2
        - SRL 最小（只有 ARG0 或 ARG1）: +1
        - ARGM-TMP 时间状语: +1（上限 1）
        - NER 命名实体: +1
        - POS 信息密度 >= 0.4: +1
        - token 数 > 8: +1
        """
        if not text or not text.strip():
            return FilterResult(should_embed=False, score=0, coherence=0, reason="empty")

        # 保留原始文本（用于关系宣言检测）
        raw_text = text.strip()

        # 预处理：去掉括号动作 + normalize
        text = strip_actions(raw_text)
        text = normalize_text(text)

        if not text or len(text) < 2:
            return FilterResult(should_embed=False, score=0, coherence=0, reason="action_only")

        # ========== 快速过滤 ==========

        # 低价值模式
        if LOW_VALUE_PATTERN.match(text):
            return FilterResult(should_embed=False, score=0, coherence=0.5, reason="low_value")

        # 纯 emoji
        if EMOJI_ONLY_PATTERN.match(text):
            return FilterResult(should_embed=False, score=0, coherence=0.3, reason="emoji_only")

        # 关系宣言句（在 strip 前后都检测）
        has_relationship = (
            any(kw in raw_text for kw in RELATIONSHIP_KEYWORDS) or
            any(kw in text for kw in RELATIONSHIP_KEYWORDS)
        )
        if has_relationship:
            # 需要有事实锚点才通过
            has_fact = FACT_ANCHOR_PATTERN.search(raw_text) or FACT_ANCHOR_PATTERN.search(text)
            if not has_fact:
                return FilterResult(should_embed=False, score=0, coherence=0.5, reason="relationship_no_fact")

        # ========== 角色门禁：assistant 更硬 ==========
        if role == "assistant":
            # assistant 默认不 embed，除非有高价值特征
            if not HIGH_VALUE_FAST_PATTERN.search(text):
                return FilterResult(should_embed=False, score=0, coherence=0.3, reason="assistant_skip")

        # ========== HanLP 分析 ==========
        try:
            analysis = self._analyze(text)
        except Exception as e:
            logger.warning(f"HanLP failed: {e}")
            # fallback 保守策略（assistant 直接拒绝）
            if role == "assistant":
                return FilterResult(should_embed=False, score=0, coherence=0.3, reason="fallback_assistant_reject")
            # user: 有高价值特征才通过
            if HIGH_VALUE_FAST_PATTERN.search(text):
                return FilterResult(
                    should_embed=True,
                    score=3,
                    coherence=0.5,
                    features=["fallback_high_value"],
                    reason="fallback_high_value"
                )
            return FilterResult(should_embed=False, score=0, coherence=0.3, reason="fallback_reject")

        score = 0
        features = []
        tokens = analysis.get("tokens", [])

        # 展平 tokens（可能是嵌套列表）
        flat_tokens = self._flatten_list(tokens)

        # ========== 1. SRL 语义角色 ==========
        srl_score, srl_features, has_tmp = self._score_srl(analysis.get("srl", []))
        score += srl_score
        features.extend(srl_features)

        # ARGM-TMP 作为加分项（上限 +1）
        if has_tmp and srl_score > 0:
            score += 1
            features.append("has_time")

        # ========== 2. NER 命名实体 ==========
        ner_list = analysis.get("ner", [])
        if self._has_ner(ner_list):
            score += 1
            features.append("has_ner")

        # ========== 3. POS 信息密度（排除功能词） ==========
        pos_list = analysis.get("pos", [])
        density = self._calc_content_density(pos_list)
        if density >= 0.4:
            score += 1
            features.append(f"density:{density:.2f}")
        elif density < 0.15 and len(flat_tokens) > 3:
            # 信息密度太低，扣分
            score -= 1
            features.append(f"low_density:{density:.2f}")

        # ========== 4. token 数加分（改用 token） ==========
        if len(flat_tokens) > 8:
            score += 1
            features.append(f"tokens:{len(flat_tokens)}")

        # ========== 5. 最终判断 ==========
        # assistant 走到这里说明命中了 HIGH_VALUE_FAST_PATTERN，用更高阈值
        threshold = 4 if role == "assistant" else 3
        should_embed = score >= threshold
        coherence = min(1.0, max(0.2, srl_score / 3.0 + density * 0.3))

        reason = "pass" if should_embed else "low_score"

        return FilterResult(
            should_embed=should_embed,
            score=score,
            coherence=coherence,
            sentence_type=None,
            elements=[],
            features=features,
            penalties=[],
            reason=reason
        )

    def _analyze(self, text: str) -> dict:
        """调用 HanLP"""
        nlp = get_hanlp()
        result = nlp(text)
        return {
            "tokens": result.get("tok/fine", []),
            "pos": result.get("pos/ctb", []),
            "ner": result.get("ner/msra", []),
            "srl": result.get("srl", []),
        }

    def _flatten_list(self, lst: list) -> list:
        """展平嵌套列表"""
        if not lst:
            return []
        if isinstance(lst[0], list):
            result = []
            for item in lst:
                result.extend(item)
            return result
        return lst

    def _score_srl(self, srl_results: list) -> tuple:
        """
        基于 SRL 打分（取最高分谓词，不累加）

        返回：(分数, 特征列表, 是否有时间状语)
        """
        if not srl_results:
            return (0, ["no_srl"], False)

        max_score = 0
        best_predicate = None
        features = []
        has_tmp = False

        for pred_info in srl_results:
            if not pred_info or len(pred_info) < 2:
                continue

            predicate = pred_info[0]
            args = pred_info[1] if len(pred_info) > 1 else []

            if not predicate:
                continue

            roles: Set[str] = set()
            for arg in args:
                if isinstance(arg, (list, tuple)) and len(arg) >= 2:
                    role = arg[0]
                    roles.add(role)
                    if role == "ARGM-TMP":
                        has_tmp = True

            # 打分（ARGM-TMP 不算 ARG0）
            has_arg0 = "ARG0" in roles
            has_arg1 = "ARG1" in roles

            # 计算这个谓词的分数
            if has_arg0 and has_arg1:
                pred_score = 3
                pred_type = "srl_complete"
            elif has_arg1:
                pred_score = 2
                pred_type = "srl_partial"
            elif has_arg0:
                pred_score = 1
                pred_type = "srl_arg0_only"
            else:
                pred_score = 0
                pred_type = "srl_pred_only"

            # 取最高分（不累加）
            if pred_score > max_score:
                max_score = pred_score
                best_predicate = predicate

            features.append(f"{pred_type}:{predicate}")

        if best_predicate and max_score > 0:
            features.insert(0, f"best_srl:{best_predicate}({max_score})")

        return (max_score, features, has_tmp)

    def _has_ner(self, ner_list: list) -> bool:
        """检查是否有命名实体"""
        if not ner_list:
            return False
        for entities in ner_list:
            if isinstance(entities, list) and len(entities) > 0:
                return True
        return False

    def _calc_content_density(self, pos_list: list) -> float:
        """
        计算内容词密度（排除功能词）

        密度 = 内容词数 / (总词数 - 功能词数)
        """
        if not pos_list:
            return 0.5

        # 展平
        flat_pos = self._flatten_list(pos_list)

        if not flat_pos:
            return 0.5

        # 统计
        content_count = sum(1 for p in flat_pos if p in CONTENT_POS_TAGS)
        function_count = sum(1 for p in flat_pos if p in FUNCTION_POS_TAGS)
        effective_total = len(flat_pos) - function_count

        if effective_total <= 0:
            return 0.5

        return content_count / effective_total

    # ========== 兼容旧接口 ==========

    def check_three_elements(self, _text: str) -> List[str]:
        return []

    def check_sentence_structure(self, _text: str) -> tuple:
        return (None, 0)

    def check_coherence(self, _text: str) -> float:
        return 0.5

    def analyze_with_hanlp(self, text: str) -> dict:
        return self._analyze(text)
