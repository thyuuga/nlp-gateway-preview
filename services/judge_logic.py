import re
from pydantic import BaseModel

class JudgeResp(BaseModel):
    label: str
    confidence: float
    reason: str

NEG = [
    (re.compile(r"不行|不可以|不能|拒绝|算了|不去"), "reject"),
    (re.compile(r"改天|下次|以后再说|到时候再看|再看看|看情况"), "defer"),
    (re.compile(r"不确定|不一定|也许|可能"), "defer"),
    (re.compile(r"(不过|但是).{0,12}(不行|不可以|不能|算了|改天|下次|以后再说|不确定|不一定|也许|可能)"), "defer"),
    (re.compile(r"真的吗[？?]?|认真的吗[？?]?|你确定吗[？?]?|能实现吗[？?]?|会实现吗[？?]?|说到做到吗[？?]?|真的假的[？?]?"), "defer"),
]
AFFIRM_STRONG = re.compile(r"约好了|说定了|一言为定|就这么定了|当然可以|没问题|行的|可以的|好呀|好的|好啊|OK", re.IGNORECASE)
AFFIRM_ACTION = re.compile(r"我陪你|我陪你去|我跟你去|我跟你一起去|一起去|我们去|我会陪你|我会记住|我记下了|我会提醒|我等着|那我等着|我会期待|期待着|记在心里|我记住了|那就这么定|就这么定|说好了|约定好了|我们说好了|那就.{0,10}(去|回去|做|玩|看|吃)")
# 新增: "那就...去/回去/做/玩/看/吃" 模式，匹配 "嗯，那就今年夏天也回去" 这种接受表达
CONDITIONAL = re.compile(r"如果|要是|只要")

def judge_acceptance(user_text: str, assistant_text: str, trace_id: str, logger=None) -> JudgeResp:
    if logger:
        logger.info("[traceId=%s] user: %s ; assistant: %s", trace_id, user_text, assistant_text)

    a = (assistant_text or "").strip()
    if not a:
        if logger:
            logger.debug("[traceId=%s] empty assistant_text", trace_id)
        return JudgeResp(label="reject", confidence=0.2, reason="empty assistant_text")

    for p, lab in NEG:
        if p.search(a):
            if logger:
                logger.debug("[traceId=%s] NEG hit: %s -> %s", trace_id, p.pattern, lab)
            return JudgeResp(label=lab, confidence=0.9, reason=f"matched:{p.pattern}")

    score, reasons = 0, []
    if AFFIRM_STRONG.search(a):
        score += 3
        reasons.append("strong_affirm")
    if AFFIRM_ACTION.search(a):
        score += 2
        reasons.append("action_affirm")

    conditional = bool(CONDITIONAL.search(a))
    if conditional and AFFIRM_ACTION.search(a):
        score = max(score, 2)
        reasons.append("conditional")

    if logger:
        logger.debug("[traceId=%s] score=%d reasons=%s", trace_id, score, reasons)

    if score >= 3:
        return JudgeResp(label="accept", confidence=0.85, reason=";".join(reasons) or "score>=3")
    if score >= 2:
        return JudgeResp(
            label="conditional_accept" if conditional else "accept",
            confidence=0.65,
            reason=";".join(reasons) or "score>=2",
        )
    return JudgeResp(label="reject", confidence=0.45, reason="no strong signals")
