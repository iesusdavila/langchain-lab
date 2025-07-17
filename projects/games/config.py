GAME_CATEGORIES = [
    "mmo", "mmorpg", "shooter", "strategy", "moba", "racing", "sports",
    "social", "sandbox", "open-world", "survival", "pvp", "pve",
    "pixel", "voxel", "zombie", "turn-based", "first-person", "third-person",
    "top-down", "tank", "space", "sailing", "side-scroller", "superhero",
    "permadeath", "card", "battle-royale", "mmo", "mmofps", "mmotps",
    "3d", "2d", "anime", "fantasy", "sci-fi", "fighting", "action-rpg",
    "action", "military", "martial-arts", "flight", "low-spec", "tower-defense",
    "horror", "mmorts"
]

GAME_PLATFORMS = [
    "pc", "browser", "web-browser"
]

SORT_CRITERIA = [
    "release-date", "alphabetical", "relevance"
]

API_CONFIG = {
    "base_url": "https://www.freetogame.com/api",
    "timeout": 30,
    "max_retries": 3
}

LLM_CONFIG = {
    "temperature": 0.7,
    "model_name": "llama3-8b-8192",
    "max_tokens": 2048
}

RESPONSE_LIMITS = {
    "games_preview": 20,
    "games_by_category": 15,
    "games_by_platform": 15,
    "sorted_games": 15,
    "description_length": 100
}
