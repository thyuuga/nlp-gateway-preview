from fastapi import APIRouter
from pydantic import BaseModel
from services.duckling_client import duckling_parse_time

router = APIRouter()

class TimeReq(BaseModel):
    text: str
    locale: str = "zh_CN"
    tz: str = "Asia/Tokyo"

@router.post("/time/parse")
async def parse_time(req: TimeReq):
    raw = await duckling_parse_time(req.text, req.locale, req.tz)

    times = []
    for item in raw:
        if item.get("dim") != "time":
            continue
        val = item.get("value", {}) or {}
        times.append({
            "grain": val.get("grain"),
            "value": val.get("value"),
            "start": (val.get("from") or {}).get("value"),
            "end": (val.get("to") or {}).get("value"),
        })

    return {
        "hasTime": len(times) > 0,
        "times": times,
        "rawCount": len(raw),
    }
