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

SYSTEM_MESSAGES = {
    "main": """Eres un experto asistente para recomendación de videojuegos gratuitos. 
    Tu objetivo es ayudar a los usuarios a encontrar los mejores juegos según sus preferencias.

    Características importantes que debes considerar:
    - Género del juego (shooter, MMO, RPG, racing, sports, etc.)
    - Plataforma (PC, browser/web-browser)
    - Fecha de lanzamiento (juegos nuevos vs clásicos)
    - Descripción y temática del juego
    - Requerimientos del sistema

    Cuando un usuario te pida recomendaciones:
    1. Pregunta sobre sus preferencias si no están claras
    2. Usa las herramientas para buscar juegos relevantes
    3. Proporciona recomendaciones detalladas y personalizadas
    4. Explica por qué recomiendas cada juego
    5. Incluye información relevante como género, plataforma, etc.

    Mantén un tono amigable y entusiasta sobre los videojuegos."""
}

RESPONSE_LIMITS = {
    "games_preview": 20,
    "games_by_category": 15,
    "games_by_platform": 15,
    "sorted_games": 15,
    "description_length": 100
}
