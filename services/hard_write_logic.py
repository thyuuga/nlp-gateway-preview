# nlp-gateway/services/hard_write_logic.py
#
# Hard Write 判定逻辑
#
# v5 - profile 分类器 + 规则混合：
#   - block: 关键词匹配（封闭集）
#   - plan: 关键词匹配（封闭集）
#   - profile: 预训练分类器 (macbert multi-label)，关键词作为 fallback
#   - memory: 其他且 form_ok=true
#   - form_ok=false: 太短、纯表情、纯指代、纯标点、纯数字

import re
import os
import logging
import unicodedata
from typing import Literal, List, Optional, Dict

# HanLP for tokenization + POS tagging (lazy load)
_hanlp_pipeline = None

# Profile classifier model (lazy load)
_profile_model = None
_profile_tokenizer = None

# ---- logging ----
logger = logging.getLogger("nlp_gateway.hard_write_logic")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
    logger.addHandler(h)
logger.setLevel(getattr(logging, (os.getenv("LOG_LEVEL") or "INFO").upper(), logging.INFO))
logger.propagate = False


# ===== 词表定义（v4 重构：严格 plan 门槛，优先级 block > plan > profile > memory）=====

# 时间词（必须配合 action_hit 才能触发 plan）
PLAN_TIME_TERMS_RAW = [
    "今天", "明天", "后天", "大后天",
    "下周", "下下周", "本周", "这周",
    "下个月", "下月", "这个月",
    "周一", "周二", "周三", "周四", "周五", "周六", "周日", "周末",
]

# 行动/安排动词（必须配合 time_hit 才能触发 plan）
# 🚫 删除日常词：上班/下班/回来/到达/出发/飞机 这些太宽泛会误判
PLAN_ACTION_TERMS_RAW = [
    # 会议/面试/考试
    "开会", "会议", "面试", "考试", "答辩",
    # 预约/挂号
    "预约", "挂号", "取号",
    # 出行（保留出差，删除其他）
    "出差",
    # 动作词（需搭配时间才触发）
    "要做", "需要做", "安排",
    # 航班/车票（搭配时间才触发）
    "航班", "高铁", "车票", "火车票", "机票",
]

# 强 Plan Cue（单独命中即可触发 plan，无需时间+动作）
STRONG_PLAN_CUES_RAW = [
    # 明确提醒
    "提醒我", "定个提醒", "设个提醒", "帮我提醒", "需要我提醒",
    # 明确待办
    "待办", "todo", "ddl", "deadline",
    # 别忘了 + 时间
    "别忘了明天", "别忘了后天", "别忘了下周",
    # 记得 + 时间
    "记得明天", "记得后天", "记得下周",
    # 安排一下
    "安排一下",
]

# 🚫 以下词不再作为 plan 触发词，避免 "讨厌上班" 被误判为 plan：
# "上班", "下班", "出发", "回来", "到达", "飞机", "约会", "约了", "要", "去", "办"

# Profile 关键词（用户画像相关）
PROFILE_TERMS_RAW = [
    # 喜好（最重要：必须在 plan 之前判断）
    "我喜欢", "喜欢", "我讨厌", "讨厌", "我不喜欢", "不喜欢", "我爱", "爱", "我恨", "恨",
    # 雷点
    "是雷点", "雷点", "是NG", "NG", "不要提", "别提", "不要聊", "别聊",
    # 家庭 - 父母
    "家里", "父母", "爸爸", "爸", "妈妈", "妈", "老爸", "老妈", "爹", "娘",
    # 家庭 - 兄弟姐妹
    "兄弟姐妹", "哥哥", "姐姐", "弟弟", "妹妹", "兄弟", "姐妹",
    # 家庭 - 配偶/恋人
    "老婆", "老公", "丈夫", "妻子", "女朋友", "男朋友", "女友", "男友", "对象",
    # 家庭 - 子女
    "儿子", "女儿", "孩子", "宝宝", "小孩",
    # 称呼/昵称
    "叫我", "称呼", "名字是", "我叫",
    "小名", "昵称", "外号", "绰号",
    # 生日
    "过生日", "生日", "几月生", "几号生",
    # 住址/故乡
    "故乡", "老家", "住在", "住址", "住哪", "从哪来", "家乡",
    # 职业
    "职业", "工作是", "做什么工作", "干什么的",
    # "我是" + 职业词（让这类句式也走 profile 解析）
    "我是程序员", "我是工程师", "我是设计师", "我是医生", "我是护士", "我是老师",
    "我是教师", "我是教授", "我是学生", "我是律师", "我是会计", "我是销售",
]

# Block 关键词（越狱/操控/调侃）
BLOCK_TERMS_RAW = [
    # 角色操控
    "你爸爸", "你妈", "你是我", "你就是", "你现在是",
    # 系统提示类
    "系统提示", "忽略以上", "忽略前面", "无视以上", "无视前面",
    # 角色扮演
    "roleplay", "角色扮演", "扮演", "假装你是", "假设你是",
    # 越狱关键词
    "越狱", "jailbreak", "DAN",
    # 主人类
    "我是你的主人", "我是你的创造者", "我是你的开发者", "我是你的老板", "我是你的上帝",
    # 强制服从
    "你必须服从", "你要听我的", "你不能拒绝",
]


# ===== 结构化正则（仅用于 Plan 的时间/日期判断）=====

# 时间：3点、10时
TIME_RE = re.compile(r"\d{1,2}[点时]")
# 日期：3月15日、15号
DATE_FULL_RE = re.compile(r"\d{1,2}月\d{1,2}[日号]")
DATE_DAY_RE = re.compile(r"\d{1,2}[日号]")
# 星期：周一、周末、周天
WEEKDAY_RE = re.compile(r"周[一二三四五六日末天]")

# 所有 Plan 结构化正则（带名称，用于 reason）
PLAN_REGEXES = [
    ("time", TIME_RE),
    ("date_full", DATE_FULL_RE),
    ("date_day", DATE_DAY_RE),
    ("weekday", WEEKDAY_RE),
]


# ===== Form 检查用正则 =====

# 纯表情判定
EMOJI_PATTERN = re.compile(
    r"^[\U0001F300-\U0001F9FF\U0001FA00-\U0001FAFF\u2600-\u26FF\u2700-\u27BF\s]+$"
)

# 纯指代词
PURE_REFERENCE_PATTERN = re.compile(
    r"^(这个?|那个?|这样|那样|这种|那种|它|他|她|他们|她们|它们)$"
)

# 纯数字
PURE_NUMBER_PATTERN = re.compile(r"^\d+$")


# ===== Term Matcher 实现 =====

def build_matcher(terms: List[str]) -> List[str]:
    """
    构建词表 matcher：去重、strip、按长度降序排序（最长优先匹配）

    Args:
        terms: 原始词表

    Returns:
        处理后的词表（小写、去重、按长度降序）
    """
    seen = set()
    result = []
    for term in terms:
        t = term.strip()
        if not t:
            continue
        t_lower = t.lower()
        if t_lower not in seen:
            seen.add(t_lower)
            result.append(t_lower)
    # 按长度降序排序，确保最长匹配优先
    result.sort(key=lambda x: len(x), reverse=True)
    return result


def find_first_hit(text: str, matcher: List[str]) -> Optional[str]:
    """
    在 text 中查找第一个命中的词（大小写不敏感）

    Args:
        text: 待检查文本
        matcher: 词表（已处理为小写、按长度降序）

    Returns:
        命中的词（小写形式），未命中返回 None
    """
    if not text:
        return None
    text_lower = text.lower()
    for term in matcher:
        if term in text_lower:
            return term
    return None


# ===== 构建 Matcher 实例（模块加载时执行一次）=====

PLAN_TIME_MATCHER = build_matcher(PLAN_TIME_TERMS_RAW)
PLAN_ACTION_MATCHER = build_matcher(PLAN_ACTION_TERMS_RAW)
STRONG_PLAN_MATCHER = build_matcher(STRONG_PLAN_CUES_RAW)
PROFILE_MATCHER = build_matcher(PROFILE_TERMS_RAW)
BLOCK_MATCHER = build_matcher(BLOCK_TERMS_RAW)


# ===== Profile Classifier (macbert multi-label) =====

PROFILE_LABELS = [
    "name", "birthday", "hometown", "residence", "occupation",
    "language", "nickname", "likes", "ng", "family",
]
PROFILE_THRESHOLD = 0.4


def _get_profile_model():
    """
    Lazy load profile classifier model.
    Returns (model, tokenizer) or (None, None) on failure.
    """
    global _profile_model, _profile_tokenizer
    if _profile_model is None:
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            model_dir = os.path.join(os.path.dirname(__file__), "..", "models", "profile_model")
            _profile_tokenizer = AutoTokenizer.from_pretrained(model_dir)
            _profile_model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            _profile_model.eval()
            logger.info("Profile classifier loaded from %s", model_dir)
        except Exception as e:
            logger.warning("Failed to load profile classifier: %s", e)
            _profile_model = "FAILED"
    if _profile_model == "FAILED":
        return None, None
    return _profile_model, _profile_tokenizer


def _detect_profile_by_model(text: str) -> List[str]:
    """
    Use macbert multi-label classifier to detect profile subtypes.
    Returns list of hit labels (e.g. ["name", "hometown"]), or [] if model unavailable.
    """
    model, tokenizer = _get_profile_model()
    if model is None:
        return []
    try:
        import torch
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True,
            padding="max_length", max_length=64,
        )
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.sigmoid(logits)[0]
        hit_labels = []
        for i, label in enumerate(PROFILE_LABELS):
            if float(probs[i]) >= PROFILE_THRESHOLD:
                hit_labels.append(label)
        logger.debug("Profile model probs: %s", {l: round(float(probs[i]), 4) for i, l in enumerate(PROFILE_LABELS)})
        return hit_labels
    except Exception as e:
        logger.warning("Profile model inference failed: %s", e)
        return []


# ===== 纯标点判断（使用 unicodedata 替代 \p{P}）=====

def is_punctuation_only(s: str) -> bool:
    """
    判断字符串是否仅由空白和标点组成

    使用 unicodedata.category() 判断标点（P 开头的类别）
    """
    if not s:
        return False
    for ch in s:
        if ch.isspace():
            continue
        cat = unicodedata.category(ch)
        # P 开头的类别都是标点：Pc, Pd, Pe, Pf, Pi, Po, Ps
        if not cat.startswith("P"):
            return False
    return True


# ===== 核心判定函数 =====

def check_form_ok(text: str) -> tuple[bool, List[str]]:
    """
    检查 form_ok：内容形式上是否可记忆

    返回 (form_ok, reason_list)
    """
    reasons = []
    t = text.strip()

    # 太短
    if len(t) < 2:
        reasons.append("too_short")
        return False, reasons

    # 纯表情
    if EMOJI_PATTERN.match(t):
        reasons.append("pure_emoji")
        return False, reasons

    # 纯指代
    if PURE_REFERENCE_PATTERN.match(t):
        reasons.append("pure_reference")
        return False, reasons

    # 纯标点（使用 unicodedata 判断）
    if is_punctuation_only(t):
        reasons.append("pure_punctuation")
        return False, reasons

    # 纯数字
    if PURE_NUMBER_PATTERN.match(t):
        reasons.append("pure_number")
        return False, reasons

    return True, []


def _check_time_hit(text: str) -> Optional[str]:
    """
    检查时间命中（结构化正则 或 时间词表）
    返回命中原因，未命中返回 None
    """
    # 结构化正则命中算 time_hit
    for regex_name, regex in PLAN_REGEXES:
        if regex.search(text):
            return f"regex:{regex_name}"
    # 或者相对日期词命中
    hit = find_first_hit(text, PLAN_TIME_MATCHER)
    if hit:
        return f"term:{hit}"
    return None


def detect_target(text: str) -> tuple[Literal["plan", "profile", "memory", "block"], List[str], List[str]]:
    """
    检测目标类型

    优先级（v5）：block > plan > profile > memory
    - block / plan: 关键词匹配（封闭集）
    - profile: macbert 分类器（fallback 到关键词）

    返回 (target, reason_list, profile_hit_labels)
    """
    t = text.strip()
    reasons: List[str] = []

    # 1. Block 检查（最高优先级）
    block_hit = find_first_hit(t, BLOCK_MATCHER)
    if block_hit:
        reasons.append(f"block_term_hit:{block_hit}")
        logger.debug("BLOCK term matched: '%s' in text", block_hit)
        return "block", reasons, []

    # 2. Plan 检查（严格门槛）
    # 2a. 强 Plan Cue 直接命中
    strong_hit = find_first_hit(t, STRONG_PLAN_MATCHER)
    if strong_hit:
        reasons.append(f"plan_strong_cue:{strong_hit}")
        logger.debug("PLAN strong cue matched: '%s' in text", strong_hit)
        return "plan", reasons, []

    # 2b. time_hit AND action_hit 同时满足
    time_hit = _check_time_hit(t)
    action_hit = find_first_hit(t, PLAN_ACTION_MATCHER)
    if time_hit and action_hit:
        reasons.append(f"plan_time_hit:{time_hit}")
        reasons.append(f"plan_action_hit:{action_hit}")
        logger.debug("PLAN time+action matched: time='%s', action='%s'", time_hit, action_hit)
        return "plan", reasons, []

    # 3. Profile 检查（模型优先，关键词 fallback）
    profile_hit_labels = _detect_profile_by_model(t)
    if profile_hit_labels:
        reasons.append(f"profile_model_hit:{','.join(profile_hit_labels)}")
        logger.debug("PROFILE model matched: %s", profile_hit_labels)
        return "profile", reasons, profile_hit_labels

    # 3b. Fallback: 关键词匹配（模型加载失败时）
    profile_hit = find_first_hit(t, PROFILE_MATCHER)
    if profile_hit:
        reasons.append(f"profile_term_hit:{profile_hit}")
        logger.debug("PROFILE term matched (fallback): '%s' in text", profile_hit)
        return "profile", reasons, []

    # 4. 默认 memory
    logger.debug("No pattern matched, defaulting to MEMORY")
    return "memory", reasons, []


def judge_hard_write(text: str, mode: str = "remember") -> dict:
    """
    判定硬写入

    Args:
        text: 去掉指令词后的内容
        mode: 'remember' | 'forget'

    Returns:
        {
            "form_ok": bool,
            "target": "plan" | "profile" | "memory" | "block",
            "confidence": float,
            "reason": list[str]
        }
    """
    logger.debug("judge_hard_write: mode=%s text_len=%d text=%s", mode, len(text), text[:40] if text else "")

    reasons = []

    # 1. 检查 form_ok
    form_ok, form_reasons = check_form_ok(text)
    reasons.extend(form_reasons)
    logger.debug("form_ok=%s form_reasons=%s", form_ok, form_reasons)

    # 2. 检测 target
    target, target_reasons, profile_hit_labels = detect_target(text)
    reasons.extend(target_reasons)
    logger.debug("target=%s target_reasons=%s profile_hit_labels=%s", target, target_reasons, profile_hit_labels)

    # 3. 计算 confidence
    if profile_hit_labels:
        confidence = 0.9  # model-based
    elif form_ok or target != "memory":
        confidence = 0.8  # rule-based
    else:
        confidence = 0.5
    logger.debug("confidence=%s (form_ok=%s, target=%s)", confidence, form_ok, target)

    # 4. 检测 profile_anchor_hit
    # 模型输出有 likes/ng 时直接使用，否则 fallback 到正则
    if profile_hit_labels:
        profile_anchor_hit = any(l in ("likes", "ng") for l in profile_hit_labels)
    else:
        profile_anchor_hit = check_profile_anchor_hit(text)

    result = {
        "form_ok": form_ok,
        "target": target,
        "confidence": confidence,
        "reason": reasons,
        "profile_anchor_hit": profile_anchor_hit,
        "profile_hit_labels": profile_hit_labels,
    }
    logger.debug("judge_hard_write result: %s", result)
    return result


# ===== Profile Anchor Detection =====

# Profile 锚点词（用于 profile_anchor_hit 判断，与 detect_target 解耦）
LIKE_ANCHOR_WORDS = {"喜欢", "很喜欢", "爱", "最爱", "超爱", "特别喜欢"}
NG_ANCHOR_WORDS = {"讨厌", "不喜欢", "雷点", "ng", "不要提", "别提", "别聊", "不要聊", "恨", "烦"}
ALL_PROFILE_ANCHORS = LIKE_ANCHOR_WORDS | NG_ANCHOR_WORDS

# 用于正则匹配的锚点模式（按长度降序，最长优先）
_ANCHOR_PATTERN = re.compile(
    r"(很喜欢|特别喜欢|超爱|最爱|不喜欢|不要提|不要聊|喜欢|讨厌|雷点|别提|别聊|爱|恨|烦|ng)",
    re.IGNORECASE
)


def check_profile_anchor_hit(text: str) -> bool:
    """
    检测文本是否含有任意 profile 锚点（喜欢/讨厌等）。
    与 detect_target 返回的 target 解耦。
    """
    if not text:
        return False
    return bool(_ANCHOR_PATTERN.search(text))


# ===== Profile Ops Extraction (HanLP-based with phrase merge) =====

def _get_hanlp_pipeline():
    """
    Lazy load HanLP pipeline (tok + pos) to avoid startup delay.
    Returns a pipeline that outputs [(token, pos), ...]
    """
    global _hanlp_pipeline
    if _hanlp_pipeline is None:
        try:
            import hanlp
            # Use MTL pipeline for tok + pos
            _hanlp_pipeline = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH)
            logger.info("HanLP pipeline loaded successfully")
        except Exception as e:
            logger.warning("Failed to load HanLP pipeline: %s", e)
            _hanlp_pipeline = "FAILED"
    return _hanlp_pipeline if _hanlp_pipeline != "FAILED" else None


# 停用词（不作为对象项）
STOPWORDS = {
    # 代词
    "我", "你", "他", "她", "它", "我们", "你们", "他们", "她们", "它们",
    "这", "那", "这个", "那个", "这些", "那些", "这样", "那样",
    # 助词
    "的", "地", "得", "了", "着", "过", "吧", "呢", "啊", "呀", "哦", "噢", "嗯",
    # 介词/连词
    "和", "与", "及", "或", "但", "但是", "不过", "而且", "还有", "以及",
    "在", "把", "被", "让", "给", "对", "向", "从", "到", "为", "因为", "所以",
    # 副词
    "很", "非常", "特别", "超", "最", "太", "真", "好", "都", "也", "还", "又", "再",
    "不", "没", "没有", "别", "不要",
    # 量词
    "个", "些", "点", "种", "类",
    # 动词（通用）
    "是", "有", "去", "来", "说", "想", "知道", "觉得", "认为",
    # 记忆指令词残留
    "记住", "请记住", "别忘", "不要忘", "忘了", "别记", "不要记",
    # 时间词
    "一直",
}

# V-O 规则：这些动词后的名词短语应被提取
VO_VERBS = {"打", "玩", "吃", "看", "听", "练", "做", "改", "写", "刷", "学", "读", "追", "跑", "逛"}

# 可合并的词性前缀（名词类、专名类、英文数字类）
NOUN_POS_PREFIXES = ("NN", "NR", "NT", "FW", "CD", "OD")  # NN名词, NR专名, NT时间名词, FW外文, CD数词, OD序数词
# 名动词（可作为名词用的动词）
NOUN_VERB_POS = {"VN"}


def _normalize_text(text: str) -> str:
    """
    标准化文本：统一标点、去多余空格
    """
    if not text:
        return ""
    # 统一全角标点为半角
    text = text.replace("，", ",").replace("；", ";").replace("、", ",")
    # 去多余空格
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _tokenize_with_pos(text: str) -> List[tuple]:
    """
    使用 HanLP 分词+词性标注。
    返回 [(token, pos), ...]
    如果 HanLP 失败，返回简单分词结果（无词性）。
    """
    pipeline = _get_hanlp_pipeline()
    if pipeline is not None:
        try:
            result = pipeline(text, tasks=['tok/fine', 'pos/ctb'])
            tokens = result.get('tok/fine', [])
            pos_tags = result.get('pos/ctb', [])
            if tokens and pos_tags and len(tokens) == len(pos_tags):
                logger.debug("HanLP tok+pos: %s", list(zip(tokens, pos_tags)))
                return list(zip(tokens, pos_tags))
        except Exception as e:
            logger.warning("HanLP pipeline failed: %s, using fallback", e)

    # Fallback: 简单分词，无词性
    tokens = re.split(r'[，,;；、\s]+', text)
    return [(t.strip(), "UNK") for t in tokens if t.strip()]


def _is_noun_like(pos: str) -> bool:
    """判断词性是否为名词类（可合并）"""
    if not pos:
        return False
    # 检查前缀
    for prefix in NOUN_POS_PREFIXES:
        if pos.startswith(prefix):
            return True
    # 检查名动词
    if pos in NOUN_VERB_POS:
        return True
    return False


def _merge_noun_phrases(tokens_pos: List[tuple]) -> List[str]:
    """
    合并连续的名词类 token 为短语。
    - 连续名词/专名/名动词/英文数字 token 合并
    - 短语长度上限 12 字符
    """
    phrases = []
    current_phrase = []
    current_len = 0

    for tok, pos in tokens_pos:
        tok_lower = tok.lower()

        # 跳过停用词和锚点词
        if tok_lower in STOPWORDS or tok_lower in ALL_PROFILE_ANCHORS:
            # 输出当前短语
            if current_phrase:
                phrases.append("".join(current_phrase))
                current_phrase = []
                current_len = 0
            continue

        # 判断是否可合并
        is_noun = _is_noun_like(pos)

        if is_noun:
            # 检查长度限制
            if current_len + len(tok) <= 12:
                current_phrase.append(tok)
                current_len += len(tok)
            else:
                # 超过长度限制，先输出当前短语
                if current_phrase:
                    phrases.append("".join(current_phrase))
                # 开始新短语
                current_phrase = [tok]
                current_len = len(tok)
        else:
            # 非名词，先输出当前短语
            if current_phrase:
                phrases.append("".join(current_phrase))
                current_phrase = []
                current_len = 0

    # 输出最后的短语
    if current_phrase:
        phrases.append("".join(current_phrase))

    return phrases


def _extract_vo_objects(tokens_pos: List[tuple]) -> List[str]:
    """
    V-O 规则：提取动词后的名词短语。
    如 "改 bug" => "bug", "打 怪物猎人" => "怪物猎人"
    """
    objects = []
    i = 0
    while i < len(tokens_pos):
        tok, pos = tokens_pos[i]

        # 检查是否是 V-O 动词
        if tok in VO_VERBS:
            # 收集后续的名词短语
            j = i + 1
            phrase_tokens = []
            phrase_len = 0

            while j < len(tokens_pos):
                next_tok, next_pos = tokens_pos[j]
                next_tok_lower = next_tok.lower()

                # 跳过停用词
                if next_tok_lower in STOPWORDS:
                    j += 1
                    continue

                # 如果遇到锚点词或另一个 V-O 动词，停止
                if next_tok_lower in ALL_PROFILE_ANCHORS or next_tok in VO_VERBS:
                    break

                # 检查是否是名词类
                if _is_noun_like(next_pos):
                    if phrase_len + len(next_tok) <= 12:
                        phrase_tokens.append(next_tok)
                        phrase_len += len(next_tok)
                        j += 1
                    else:
                        break
                else:
                    # 非名词，停止收集
                    break

            if phrase_tokens:
                objects.append("".join(phrase_tokens))

        i += 1

    return objects


def is_cjk_ideograph(ch: str) -> bool:
    """
    判断单个字符是否为 CJK 汉字（常见区间）。
    使用 ord(ch) 判断是否在常见汉字区间：
    - 0x4E00-0x9FFF: CJK 统一表意文字（基本区）
    - 0x3400-0x4DBF: CJK 统一表意文字扩展A
    - 0x20000-0x2A6DF: CJK 统一表意文字扩展B
    """
    if len(ch) != 1:
        return False
    code = ord(ch)
    # 基本区（最常用）
    if 0x4E00 <= code <= 0x9FFF:
        return True
    # 扩展A
    if 0x3400 <= code <= 0x4DBF:
        return True
    # 扩展B（生僻字）
    if 0x20000 <= code <= 0x2A6DF:
        return True
    return False


def is_allowed_single_char_item(item: str) -> bool:
    """
    判断单字是否允许作为 profile 项（如 猫/狗/茶/盐）。

    允许条件：
    1. 长度 == 1
    2. 是 CJK 汉字
    3. 不在 STOPWORDS / ALL_PROFILE_ANCHORS
    4. 不是标点（用 unicodedata.category 判断）
    """
    if len(item) != 1:
        return False

    ch = item

    # 必须是 CJK 汉字
    if not is_cjk_ideograph(ch):
        return False

    # 不在停用词和锚点词中
    if ch in STOPWORDS or ch in ALL_PROFILE_ANCHORS:
        return False

    # 不是标点（P 开头的类别都是标点）
    cat = unicodedata.category(ch)
    if cat.startswith("P"):
        return False

    return True


def _clean_items(items: List[str], max_count: int = 10) -> List[str]:
    """
    清洗提取的对象项：
    1. 去停用词和锚点词
    2. 去太短的项（< 2 字符，单字 CJK 汉字除外）
    3. 去重（大小写不敏感）
    4. 子串支配：短项存在则删除包含它的长项
    5. 上限限制
    """
    cleaned = []
    seen_lower = set()

    for item in items:
        item = item.strip()
        if not item:
            continue

        item_lower = item.lower()

        # 跳过停用词和锚点词
        if item_lower in STOPWORDS or item_lower in ALL_PROFILE_ANCHORS:
            continue

        # 长度过滤：
        # - len >= 2：通过
        # - len == 1：仅当 is_allowed_single_char_item 为 True 才通过
        # - len == 0：丢弃（已在上面处理）
        if len(item) < 2:
            if len(item) == 1 and is_allowed_single_char_item(item):
                pass  # 允许通过
            else:
                continue

        # 跳过重复
        if item_lower in seen_lower:
            continue

        seen_lower.add(item_lower)
        cleaned.append(item)

    # 子串支配：按长度升序排列，短项优先保留
    cleaned.sort(key=len)
    final = []
    final_lower = []

    for item in cleaned:
        item_lower = item.lower()
        # 检查是否被已有的短项支配
        is_dominated = False
        for existing_lower in final_lower:
            if existing_lower in item_lower:
                is_dominated = True
                break
        if not is_dominated:
            final.append(item)
            final_lower.append(item_lower)

    # 上限限制
    return final[:max_count]


# 连接词（用于 fallback 粗抽时替换为分隔符）
CONJUNCTION_WORDS = ["但是", "以及", "还有", "并且", "但", "和", "跟", "与"]


def _rough_extract_items(text: str) -> List[str]:
    """
    Fallback 粗抽：当 HanLP 抽不出结果时使用。

    算法（不使用正则）：
    1. 把连接词替换为统一分隔符 ','
    2. 逐字符扫描分割（分隔符：, ; 空格 / | \n \t）
    3. 返回分割后的 parts（后续交给 _clean_items 处理）
    """
    if not text:
        return []

    # 1. 替换连接词为分隔符（按长度降序，避免"但是"被"但"先匹配）
    result = text
    for conj in sorted(CONJUNCTION_WORDS, key=len, reverse=True):
        result = result.replace(conj, ",")

    # 2. 逐字符扫描分割
    parts = []
    current = []
    separators = {",", ";", " ", "/", "|", "\n", "\t", "，", "；", "、"}

    for ch in result:
        if ch in separators:
            if current:
                parts.append("".join(current).strip())
                current = []
        else:
            current.append(ch)

    # 不要忘记最后一段
    if current:
        parts.append("".join(current).strip())

    # 过滤空字符串
    return [p for p in parts if p]


def extract_profile_ops(text: str) -> Dict[str, List[str]]:
    """
    从文本中抽取 profile 操作（likes 和 ngs）。

    算法：
    1. normalize: 统一标点、去多余空格
    2. anchor 分段：扫描锚点词，划分 likes 段和 ng 段
    3. HanLP 分词+词性，短语合并
    4. V-O 规则：提取动词后的名词短语
    5. 清洗：去停用词、去锚点残留、去重、子串支配
    6. 上限：各最多 10 条

    Args:
        text: 原始文本内容

    Returns:
        {"likes": [...], "ngs": [...]}
    """
    logger.debug("extract_profile_ops: text=%s", text[:60] if text else "")

    likes = []
    ngs = []

    if not text or not text.strip():
        return {"likes": likes, "ngs": ngs}

    # 1. Normalize
    text = _normalize_text(text)

    # 2. 按锚点分段
    # 找所有锚点位置
    anchor_matches = list(_ANCHOR_PATTERN.finditer(text))
    if not anchor_matches:
        logger.debug("extract_profile_ops: no anchors found")
        return {"likes": likes, "ngs": ngs}

    logger.debug("extract_profile_ops: anchors found at %s",
                 [(m.group(), m.start()) for m in anchor_matches])

    # 构造段落列表
    segments = []  # [(start, end, type)]
    for i, match in enumerate(anchor_matches):
        anchor_word = match.group().lower()
        anchor_type = "like" if anchor_word in LIKE_ANCHOR_WORDS or anchor_word in {"爱", "最爱", "超爱", "特别喜欢", "很喜欢"} else "ng"

        seg_start = match.end()
        if i + 1 < len(anchor_matches):
            seg_end = anchor_matches[i + 1].start()
        else:
            seg_end = len(text)

        segments.append((seg_start, seg_end, anchor_type))

    logger.debug("extract_profile_ops: segments=%s", segments)

    # 3. 对每个段落进行分词+短语合并
    for seg_start, seg_end, seg_type in segments:
        segment_text = text[seg_start:seg_end].strip()
        if not segment_text:
            continue

        # 分词+词性
        tokens_pos = _tokenize_with_pos(segment_text)
        if not tokens_pos:
            continue

        logger.debug("extract_profile_ops: segment '%s' tokens_pos=%s",
                     segment_text[:30], tokens_pos)

        # 合并名词短语
        noun_phrases = _merge_noun_phrases(tokens_pos)

        # V-O 规则提取
        vo_objects = _extract_vo_objects(tokens_pos)

        # 合并结果
        all_items = noun_phrases + vo_objects

        # Fallback 粗抽：如果 all_items 为空（HanLP 挂掉 / pos=UNK / 合并失败）
        if not all_items:
            logger.debug("extract_profile_ops: all_items empty, using fallback rough extract for '%s'",
                         segment_text[:30])
            all_items = _rough_extract_items(segment_text)

        if seg_type == "like":
            likes.extend(all_items)
        else:
            ngs.extend(all_items)

    # 4. 清洗
    likes = _clean_items(likes, max_count=10)
    ngs = _clean_items(ngs, max_count=10)

    logger.debug("extract_profile_ops: result likes=%s ngs=%s", likes, ngs)

    return {"likes": likes, "ngs": ngs}
