from datetime import datetime
from typing import List, Dict, Any

class GameDataProcessor:    
    @staticmethod
    def format_game_summary(game: Dict[str, Any]) -> str:
        id = game['id']
        title = game['title']
        genre = game['genre']
        platform = game['platform']
        description = game['short_description']
                
        return f"• {title} (id: {id})\n Genero: {genre}\n  Plataforma: {platform}\n  Descripción: {description}\n"
    
    @staticmethod
    def format_detailed_game(game: Dict[str, Any]) -> str:
        result = f"{game.get('title', 'N/A')}\n"
        result += "=" * (len(game.get('title', '')) + 4) + "\n\n"
        
        result += f"Género: {game.get('genre', 'N/A')}\n"
        result += f"Plataforma: {game.get('platform', 'N/A')}\n"
        result += f"Desarrollador: {game.get('developer', 'N/A')}\n"
        result += f"Editor: {game.get('publisher', 'N/A')}\n"
        result += f"Fecha de lanzamiento: {game.get('release_date', 'N/A')}\n\n"
        
        if game.get('description'):
            result += f"Descripción:\n{game['description']}\n\n"
        
        if game.get('minimum_system_requirements'):
            req = game['minimum_system_requirements']
            result += "Requerimientos mínimos:\n"
            if req.get('os'):
                result += f"OS: {req['os']}\n"
            if req.get('processor'):
                result += f"Procesador: {req['processor']}\n"
            if req.get('memory'):
                result += f"Memoria: {req['memory']}\n"
            if req.get('graphics'):
                result += f"Gráficos: {req['graphics']}\n"
            if req.get('storage'):
                result += f"Almacenamiento: {req['storage']}\n"
            result += "\n"
        
        if game.get('game_url'):
            result += f"URL del juego: {game['game_url']}\n"
        
        if game.get('screenshots'):
            result += f"Capturas de pantalla: {len(game['screenshots'])} disponibles\n"
        
        return result
    
    @staticmethod
    def filter_games_by_keyword(games: List[Dict[str, Any]], keyword: str) -> List[Dict[str, Any]]:
        keyword_lower = keyword.lower()
        filtered_games = []
        
        for game in games:
            title = game['title'].lower()
            description = game['short_description'].lower()
            
            if (keyword_lower in title or 
                keyword_lower in description):
                filtered_games.append(game)
        
        return filtered_games
    
    @staticmethod
    def get_recent_games(games: List[Dict[str, Any]], limit: int = 10) -> List[Dict[str, Any]]:
        games_with_dates = []
        
        for game in games:
            release_date = game.get('release_date')
            if release_date:
                try:
                    date_obj = datetime.strptime(release_date, '%Y-%m-%d')
                    games_with_dates.append((game, date_obj))
                except ValueError:
                    games_with_dates.append((game, datetime.min))
            else:
                games_with_dates.append((game, datetime.min))
        
        games_with_dates.sort(key=lambda x: x[1], reverse=True)
        return [game for game, _ in games_with_dates[:limit]]
    
    @staticmethod
    def create_recommendation_summary(games: List[Dict[str, Any]], criteria: str) -> str:
        if not games:
            return f"No se encontraron juegos que coincidan con el criterio: {criteria}"
        
        summary = f"Recomendaciones basadas en: {criteria}\n"
        summary += f"{len(games)} juegos encontrados\n\n"
        
        genres = GameDataProcessor.get_popular_genres(games)
        top_genres = list(genres.keys())[:3]
        
        if top_genres:
            summary += f"Géneros principales: {', '.join(top_genres)}\n\n"
        
        for i, game in enumerate(games[:5], 1):
            summary += f"{i}. {game.get('title', 'N/A')} ({game.get('genre', 'N/A')})\n"
            summary += f"   {game.get('short_description', 'Sin descripción')[:80]}...\n\n"
        
        if len(games) > 5:
            summary += f"... y {len(games) - 5} juegos más disponibles.\n"
        
        return summary

class QueryProcessor:    
    @staticmethod
    def extract_preferences(query: str) -> Dict[str, Any]:
        query_lower = query.lower()
        preferences = {
            'genres': [],
            'platforms': [],
            'keywords': [],
            'sort_preference': None
        }
        
        genre_keywords = {
            'shooter': ['shooter', 'fps', 'disparo', 'disparos'],
            'mmo': ['mmo', 'masivo', 'multijugador masivo'],
            'racing': ['carreras', 'racing', 'coches', 'autos'],
            'sports': ['deportes', 'sports', 'fútbol', 'basketball'],
            'strategy': ['estrategia', 'strategy', 'rts'],
            'rpg': ['rpg', 'rol', 'aventura'],
            'action': ['acción', 'action'],
            'puzzle': ['puzzle', 'rompecabezas'],
            'horror': ['horror', 'miedo', 'terror']
        }
        
        for genre, keywords in genre_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                preferences['genres'].append(genre)
        
        if any(word in query_lower for word in ['pc', 'computadora', 'ordenador']):
            preferences['platforms'].append('pc')
        if any(word in query_lower for word in ['browser', 'navegador', 'web']):
            preferences['platforms'].append('browser')
        
        if any(word in query_lower for word in ['nuevo', 'reciente', 'último']):
            preferences['sort_preference'] = 'release-date'
        elif any(word in query_lower for word in ['popular', 'relevante']):
            preferences['sort_preference'] = 'relevance'
        
        return preferences