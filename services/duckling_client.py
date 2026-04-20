import os
import httpx

DEFAULT_DUCKLING_URL = "http://127.0.0.1:8001/parse"

def get_duckling_url() -> str:
    return os.getenv("DUCKLING_URL") or DEFAULT_DUCKLING_URL

async def duckling_parse_time(text: str, locale: str = "zh_CN", tz: str = "Asia/Tokyo", timeout_sec: float = 3.0):
    url = get_duckling_url()
    data = {
        "text": text,
        "locale": locale,
        "tz": tz,
    }
    async with httpx.AsyncClient(timeout=timeout_sec) as client:
        resp = await client.post(
            url,
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        resp.raise_for_status()
        return resp.json()
