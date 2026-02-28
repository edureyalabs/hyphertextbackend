import httpx
from config import BRAVE_SEARCH_API_KEY, BRAVE_SEARCH_URL


async def brave_search(query: str, count: int = 5) -> list[dict]:
    if not BRAVE_SEARCH_API_KEY:
        return [{"title": "Search unavailable", "description": "No Brave Search API key configured.", "url": ""}]

    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": BRAVE_SEARCH_API_KEY
    }
    params = {
        "q": query,
        "count": count,
        "text_decorations": False,
        "search_lang": "en"
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(BRAVE_SEARCH_URL, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()

    results = []
    web_results = data.get("web", {}).get("results", [])
    for r in web_results:
        results.append({
            "title": r.get("title", ""),
            "description": r.get("description", ""),
            "url": r.get("url", "")
        })
    return results


def format_search_results(results: list[dict]) -> str:
    if not results:
        return "No results found."
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r['title']}")
        lines.append(f"   {r['description']}")
        lines.append(f"   {r['url']}")
        lines.append("")
    return "\n".join(lines)