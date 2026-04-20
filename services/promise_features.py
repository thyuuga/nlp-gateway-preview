"""
Promise 判定模块 v2

基于以下规格：
1. Promise 必须同时满足：承诺意图 + 未来指向 + 可落库内容
2. Event Promise：主语是承诺方 + 行动动词 + 未来信号
3. Relationship Promise：分 B1(行为约束型，入库) 和 B2(情绪宣告型，不入库)
4. 排除：纯问句、纯确认句、纯愿望
"""

from typing import Dict, Any, Optional, List
from datetime import datetime

from dateutil import parser as dateutil_parser

from services.lang_guess import guess_lang
from services.nlp import get_nlp
from services.duckling_client import duckling_parse_time


DUCKLING_LOCALE = {
    "zh": "zh_CN",
    "ja": "ja_JP",
    "en": "en_US",
}

# ============================================================
# 词表定义
# ============================================================

# --- 行动动词（可执行动作）---
ACTION_VERBS = [
    "去", "来", "回", "见", "做", "给", "买", "发", "还", "帮", "安排", "改",
    "吃", "喝", "玩", "逛", "看", "听", "学", "教", "送", "接", "带", "拿",
    "旅行", "出发", "参加", "约", "陪", "提醒", "准备", "开始", "完成",
]

# --- 状态动词（不算行动）---
STATIVE_VERBS = [
    "感觉", "觉得", "知道", "想", "认为", "以为", "希望", "相信", "怀疑",
    "喜欢", "爱", "恨", "怕", "担心", "在乎", "介意", "理解", "明白",
    "是", "有", "在", "像", "属于", "等于",
]

# --- 复数人/共同行动词 ---
# 这些词表示"双方共同参与"，而不只是"提到对方"
PLURAL_ACTION_MARKERS = [
    "我们", "咱们", "咱俩", "一起",
    "和你", "跟你", "陪你", "带你", "给你", "帮你",
    "和我", "跟我", "陪我", "带我", "给我", "帮我",
]

# --- 模糊未来词（Duckling 解析不了的）---
FUZZY_FUTURE = ["将来", "以后", "之后", "未来", "到时候", "改天", "有空", "下次", "下回"]

# --- 未来意图词（表达承诺意图）---
FUTURE_INTENT = ["会", "要", "打算", "准备", "一定", "肯定", "保证", "答应"]

# --- 邀约语气词（提案信号）---
INVITE_TONE = ["吧", "好吗", "要不要", "怎么样", "行吗", "可以吗"]

# --- 确认句排除（这些是对已有约定的确认，不是新提案）---
CONFIRM_EXCLUDE = ["约定好了", "约好了", "说好了", "说定了", "定好了", "就这么定", "一言为定"]

# --- 问句排除模式 ---
QUESTION_PATTERNS = ["怎么样", "好吗", "行吗", "可以吗", "是吗", "对吗", "呢", "吗"]
PURE_QUESTION_STARTERS = ["你现在", "你最近", "你今天", "你觉得", "你感觉", "你还好"]

# --- 单人自律词（排除）---
SELF_DISCIPLINE = ["健身", "努力", "自律", "学习", "减肥", "戒烟", "早起", "打卡", "锻炼"]

# --- Relationship Promise B1: 行为约束型（入库）---
REL_B1_CONSTRAINT_VERBS = [
    "不会再", "不再", "再也不", "永远不", "不会",
    "会尊重", "会坦诚", "会陪伴", "会做到", "会改",
    "不提", "不隐瞒", "不骗", "不冷暴力", "不吵架",
]
REL_B1_DURATION = ["以后", "一直", "从现在起", "从今以后", "永远"]

# --- Relationship Promise B2: 情绪宣告型（不入库）---
REL_B2_EMOTIONAL = [
    "爱你", "喜欢你", "在乎你", "珍惜你", "想你", "需要你",
    "在你身边", "守护你", "保护你", "支持你",
]


# ============================================================
# 辅助函数
# ============================================================

async def _duckling_time_async(text: str, lang: str, tz: str) -> List[Dict[str, Any]]:
    """调用 Duckling 解析时间"""
    locale = DUCKLING_LOCALE.get(lang, "zh_CN")
    try:
        return await duckling_parse_time(text, locale, tz, timeout_sec=2.5)
    except Exception:
        return []


async def _has_future_time_duckling(text: str, lang: str, tz: str) -> bool:
    """
    用 Duckling 检测是否有未来时间。
    通过比较解析出的时间戳与当前时间。
    """
    items = await _duckling_time_async(text, lang, tz)
    if not items:
        return False

    now = datetime.now()

    for it in items:
        val = it.get("value") or {}
        vtype = val.get("type")

        if vtype == "value":
            time_str = val.get("value")
            if time_str:
                try:
                    parsed_time = dateutil_parser.isoparse(time_str)
                    # 转换为 naive datetime 比较
                    parsed_naive = parsed_time.replace(tzinfo=None)
                    if parsed_naive > now:
                        return True
                except Exception:
                    pass

        elif vtype == "interval":
            # interval 检查 from 是否在未来
            from_val = val.get("from", {}).get("value")
            if from_val:
                try:
                    parsed_time = dateutil_parser.isoparse(from_val)
                    parsed_naive = parsed_time.replace(tzinfo=None)
                    if parsed_naive > now:
                        return True
                except Exception:
                    pass

    return False


def _has_fuzzy_future(text: str) -> bool:
    """检测模糊未来词（将来、以后等）"""
    return any(w in text for w in FUZZY_FUTURE)


def _has_future_intent(text: str) -> bool:
    """检测未来意图词（会、要、打算等）"""
    return any(w in text for w in FUTURE_INTENT)


async def _has_future_signal(text: str, lang: str, tz: str) -> bool:
    """
    综合检测未来信号：
    1. Duckling 解析到具体未来时间
    2. 模糊未来词
    3. 未来意图词
    """
    # 方案1：Duckling 具体未来时间
    if await _has_future_time_duckling(text, lang, tz):
        return True

    # 方案2：模糊未来词
    if _has_fuzzy_future(text):
        return True

    # 方案3：未来意图词
    if _has_future_intent(text):
        return True

    return False


def _has_action_verb(text: str, lang: str) -> bool:
    """
    检测是否有行动动词。
    使用 spaCy 检测 VERB，但排除状态动词。
    """
    nlp = get_nlp(lang)
    doc = nlp(text)

    # 用 spaCy 检测动词，但排除状态动词
    for tok in doc:
        if tok.pos_ == "VERB":
            # 检查是否是状态动词
            if tok.text not in STATIVE_VERBS and tok.lemma_ not in STATIVE_VERBS:
                # 进一步确认是行动动词
                if tok.text in ACTION_VERBS or tok.lemma_ in ACTION_VERBS:
                    return True
                # spaCy 标注为 VERB 且不是状态动词，也接受
                if tok.text not in STATIVE_VERBS:
                    return True

    # Fallback：检查行动动词列表
    if any(v in text for v in ACTION_VERBS):
        return True

    return False


def _has_plural_action(text: str) -> bool:
    """
    检测是否有复数人共同行动。
    需要"我们/一起"等共同行动标记，而不只是"你"。
    """
    return any(w in text for w in PLURAL_ACTION_MARKERS)


def _is_pure_question(text: str) -> bool:
    """
    检测是否是纯问句/关心句。
    如："你现在感觉怎么样"、"你还好吗"
    """
    # 以关心句式开头 + 没有行动动词
    for starter in PURE_QUESTION_STARTERS:
        if text.startswith(starter):
            # 检查是否只是问候，没有实际行动
            has_action = any(v in text for v in ACTION_VERBS)
            if not has_action:
                return True

    return False


def _is_confirm_without_content(text: str) -> bool:
    """
    检测是否是纯确认句但没有可落库内容。
    如："我们约定好了哦"（没有具体事项/时间/行动）
    """
    # 包含确认词
    has_confirm = any(c in text for c in CONFIRM_EXCLUDE)
    if not has_confirm:
        return False

    # 检查是否有具体内容（行动动词 + 未来词/时间）
    has_action = any(v in text for v in ACTION_VERBS)
    has_future = _has_fuzzy_future(text) or _has_future_intent(text)

    # 如果只有确认词，没有行动或未来信息，就是空确认
    if not has_action and not has_future:
        return True

    return False


def _is_self_only(text: str) -> bool:
    """
    检测是否是单人自律/自我要求。
    没有复数人标记 + 有自律词。
    """
    if _has_plural_action(text):
        return False

    # 检查自律词
    if any(k in text for k in SELF_DISCIPLINE):
        return True

    return False


def _detect_relationship_b1(text: str) -> bool:
    """
    检测 B1 行为约束型 relationship promise。
    必须有：约束动词 + 持续性词。
    """
    has_constraint = any(v in text for v in REL_B1_CONSTRAINT_VERBS)
    has_duration = any(d in text for d in REL_B1_DURATION)

    return has_constraint and has_duration


def _detect_relationship_b2(text: str) -> bool:
    """
    检测 B2 情绪宣告型 relationship promise。
    这类不入库。
    """
    return any(e in text for e in REL_B2_EMOTIONAL)


# ============================================================
# 主入口
# ============================================================

async def extract_promise_candidate_async(
    text: str,
    tz: str = "Asia/Tokyo",
    assistant_text: Optional[str] = None  # 保留参数，用于未来扩展（如检测 assistant 提案）
) -> Dict[str, Any]:
    """
    提取 promise candidate。

    返回结构：
    {
        "isCandidate": bool,
        "type": "none" | "event" | "relationship_b1" | "relationship_b2",
        "confidence": float,
        "signals": { ... },
        "meta": { "lang": ... }
    }
    """
    _ = assistant_text  # 保留参数，暂未使用
    t = (text or "").strip()
    if not t:
        return _build_result(False, "none", 0.0, {}, "zh")

    lang = guess_lang(t)

    # ========== 排除检测 ==========

    # 1. 纯问句/关心句
    if _is_pure_question(t):
        return _build_result(
            False, "none", 0.0,
            {"excluded_reason": "pure_question"},
            lang
        )

    # 2. 纯确认句但没内容
    if _is_confirm_without_content(t):
        return _build_result(
            False, "none", 0.0,
            {"excluded_reason": "confirm_without_content"},
            lang
        )

    # 3. 单人自律
    if _is_self_only(t):
        return _build_result(
            False, "none", 0.0,
            {"excluded_reason": "self_only"},
            lang
        )

    # ========== 信号检测 ==========

    has_future = await _has_future_signal(t, lang, tz)
    has_action = _has_action_verb(t, lang)
    has_plural = _has_plural_action(t)
    has_invite_tone = any(tone in t for tone in INVITE_TONE)

    # ========== Relationship Promise 检测 ==========

    is_rel_b1 = _detect_relationship_b1(t)
    is_rel_b2 = _detect_relationship_b2(t)

    if is_rel_b1:
        # B1 行为约束型 → 入库
        return _build_result(
            True, "relationship_b1", 0.85,
            {
                "hasFuture": has_future,
                "hasAction": has_action,
                "hasPlural": has_plural,
                "isRelB1": True,
                "isRelB2": False,
            },
            lang
        )

    if is_rel_b2 and not has_action:
        # B2 情绪宣告型 → 不入库（但标记类型供参考）
        return _build_result(
            False, "relationship_b2", 0.5,
            {
                "hasFuture": has_future,
                "hasAction": has_action,
                "hasPlural": has_plural,
                "isRelB1": False,
                "isRelB2": True,
                "excluded_reason": "emotional_declaration",
            },
            lang
        )

    # ========== Event Promise 检测 ==========

    # Event Promise 必须满足：
    # 1. 有行动动词
    # 2. 有复数人/共同行动
    # 3. 有未来信号（时间/意图词）或邀约语气

    if has_action and has_plural:
        if has_future or has_invite_tone:
            conf = 0.6
            if has_future:
                conf += 0.15
            if has_invite_tone:
                conf += 0.1
            conf = min(0.9, conf)

            return _build_result(
                True, "event", conf,
                {
                    "hasFuture": has_future,
                    "hasAction": has_action,
                    "hasPlural": has_plural,
                    "hasInviteTone": has_invite_tone,
                },
                lang
            )

    # ========== 不满足条件 ==========

    return _build_result(
        False, "none", 0.0,
        {
            "hasFuture": has_future,
            "hasAction": has_action,
            "hasPlural": has_plural,
            "hasInviteTone": has_invite_tone if 'has_invite_tone' in dir() else False,
            "missing": _get_missing_conditions(has_action, has_plural, has_future, has_invite_tone if 'has_invite_tone' in dir() else False),
        },
        lang
    )


def _build_result(
    is_candidate: bool,
    ptype: str,
    confidence: float,
    signals: Dict[str, Any],
    lang: str
) -> Dict[str, Any]:
    """构建返回结果"""
    return {
        "isCandidate": is_candidate,
        "type": ptype,
        "confidence": confidence,
        "signals": signals,
        "meta": {"lang": lang},
    }


def _get_missing_conditions(
    has_action: bool,
    has_plural: bool,
    has_future: bool,
    has_invite_tone: bool
) -> List[str]:
    """返回缺失的条件列表，用于调试"""
    missing = []
    if not has_action:
        missing.append("action_verb")
    if not has_plural:
        missing.append("plural_action")
    if not has_future and not has_invite_tone:
        missing.append("future_signal_or_invite_tone")
    return missing
