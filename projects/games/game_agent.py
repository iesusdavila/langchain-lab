import os
import requests
from langchain import hub
from dotenv import load_dotenv
from langchain.tools import Tool
from typing import List, Dict, Any
from langchain_groq import ChatGroq
from utils import GameDataProcessor, QueryProcessor
from langchain.agents import create_react_agent, AgentExecutor
from config import API_CONFIG, LLM_CONFIG, GAME_CATEGORIES, GAME_PLATFORMS, SORT_CRITERIA, RESPONSE_LIMITS, SYSTEM_MESSAGES

load_dotenv()

class FreeToGameAPI:
    def __init__(self):
        pass

    def _make_request(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        url = f"{API_CONFIG['base_url']}/{endpoint}"
        
        for attempt in range(API_CONFIG['max_retries']):
            try:
                response = requests.get(url, params=params, timeout=API_CONFIG["timeout"])
                response.raise_for_status()

                return response.json()
            except requests.RequestException as e:
                if attempt == API_CONFIG['max_retries'] - 1:
                    return {"error": f"Error en petición después de {API_CONFIG['max_retries']} intentos: {str(e)}"}
                continue
    
    def get_all_games(self) -> List[Dict[str, Any]]:
        return self._make_request("games")
    
    def get_game_by_id(self, game_id: int) -> Dict[str, Any]:
        return self._make_request("game", {"id": game_id})
    
    def get_games_by_category(self, category: str) -> List[Dict[str, Any]]:
        if category not in GAME_CATEGORIES:
            return {"error": f"Categoría '{category}' no válida. Categorías disponibles: {', '.join(GAME_CATEGORIES)}"}
        return self._make_request("games", {"category": category})
    
    def get_games_by_platform(self, platform: str) -> List[Dict[str, Any]]:
        if platform not in GAME_PLATFORMS:
            return {"error": f"Plataforma '{platform}' no válida. Plataformas disponibles: {', '.join(GAME_PLATFORMS)}"}
        return self._make_request("games", {"platform": platform})
    
    def get_games_sorted(self, sort_by: str) -> List[Dict[str, Any]]:
        if sort_by not in SORT_CRITERIA:
            return {"error": f"Criterio '{sort_by}' no válido. Criterios disponibles: {', '.join(SORT_CRITERIA)}"}
        return self._make_request("games", {"sort-by": sort_by})

class GameRecommendationAgent:
    
    def __init__(self):
        self.api = FreeToGameAPI()
        self.processor = GameDataProcessor()
        self.query_processor = QueryProcessor()
        self.llm = ChatGroq(
            temperature=LLM_CONFIG["temperature"],
            groq_api_key=os.getenv("GROP_API_KEY"),
            model_name=LLM_CONFIG["model_name"],
            max_tokens=LLM_CONFIG["max_tokens"]
        )
        self.tools = self._create_tools()
        self.agent = self._create_agent()
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            memory=None
        )
        self.all_games = self.api.get_all_games()
    
    def _create_tools(self) -> List[Tool]:
        
        def search_all_games(query: str) -> str:
            games = self.all_games
            if isinstance(games, dict) and "error" in games:
                return games["error"]
            
            if query and query.strip():
                filtered_games = self.processor.filter_games_by_keyword(games, query)
                if filtered_games:
                    games = filtered_games
                    search_info = f"Búsqueda filtrada por '{query}': "
                else:
                    search_info = f"No se encontraron juegos con '{query}', mostrando todos: "
            else:
                search_info = "Todos los juegos disponibles: "
            
            limited_games = games[:RESPONSE_LIMITS["games_preview"]]
            result = f"{search_info}{len(games)} juegos encontrados. Mostrando primeros {len(limited_games)}:\n\n"
            
            for game in limited_games:
                result += self.processor.format_game_summary(game)
                result += "\n"
            
            return result
        
        def search_games_by_category(category: str) -> str:
            games = self.api.get_games_by_category(category)
            if isinstance(games, dict) and "error" in games:
                return games["error"]
            
            if not games:
                available_categories = ", ".join(GAME_CATEGORIES[:10])
                return f"No se encontraron juegos en la categoría '{category}'. Categorías disponibles: {available_categories}..."
            
            result = self.processor.create_recommendation_summary(games, f"categoría '{category}'")
            
            for game in games[:RESPONSE_LIMITS["games_by_category"]]:
                result += self.processor.format_game_summary(game)
                result += "\n"
            
            return result
        
        def search_games_by_platform(platform: str) -> str:
            games = self.api.get_games_by_platform(platform)
            if isinstance(games, dict) and "error" in games:
                return games["error"]
            
            if not games:
                available_platforms = ", ".join(GAME_PLATFORMS)
                return f"No se encontraron juegos para la plataforma '{platform}'. Plataformas disponibles: {available_platforms}"
            
            result = self.processor.create_recommendation_summary(games, f"plataforma '{platform}'")
            
            for game in games[:RESPONSE_LIMITS["games_by_platform"]]:
                result += self.processor.format_game_summary(game)
                result += "\n"
            
            return result
        
        def get_game_details(game_id: str) -> str:
            try:
                game_id_int = int(game_id)
            except ValueError:
                games = self.all_games
                if isinstance(games, dict) and "error" in games:
                    return games["error"]
                
                if game_id and game_id.strip():
                    filtered_games = self.processor.filter_games_by_keyword(games, game_id)
                    if filtered_games:
                        game_id_int = filtered_games[0]['id']
                    else:
                        return "Error: No se encontraron juegos con ese ID o nombre."
                else:
                    return "Error: El ID del juego esta vacío."
            
            game = self.api.get_game_by_id(game_id_int)
            if isinstance(game, dict) and "error" in game:
                return game["error"]
            
            return self.processor.format_detailed_game(game)
        
        def get_sorted_games(sort_criteria: str) -> str:
            games = self.api.get_games_sorted(sort_criteria)
            if isinstance(games, dict) and "error" in games:
                return games["error"]
            
            result = self.processor.create_recommendation_summary(games, f"ordenamiento por '{sort_criteria}'")
            
            for game in games[:RESPONSE_LIMITS["sorted_games"]]:
                result += self.processor.format_game_summary(game)
                result += f"  📅 Fecha: {game.get('release_date', 'N/A')}\n\n"
            
            return result
        
        return [
            Tool(
                name="buscar_todos_los_juegos",
                description="Busca y muestra información general de todos los juegos disponibles. Puede filtrar por palabra clave si se proporciona.",
                func=search_all_games
            ),
            Tool(
                name="buscar_por_categoria",
                description=f"Busca juegos por categoría específica. Categorías disponibles: {', '.join(GAME_CATEGORIES[:10])}...",
                func=search_games_by_category
            ),
            Tool(
                name="buscar_por_plataforma", 
                description=f"Busca juegos por plataforma específica. Plataformas: {', '.join(GAME_PLATFORMS)}",
                func=search_games_by_platform
            ),
            Tool(
                name="obtener_detalles_juego",
                description="Obtiene detalles, información o caracteristicas de un juego específico usando su ID numérico",
                func=get_game_details
            ),
            Tool(
                name="obtener_juegos_ordenados",
                description=f"Obtiene juegos ordenados por criterio específico: {', '.join(SORT_CRITERIA)}",
                func=get_sorted_games
            )
        ]
    
    def get_prompt(self) -> str:
        return hub.pull("hwchase17/react")

    
    def _create_agent(self):
        return create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.get_prompt(),
        )
    
    def recommend_games(self, user_query: str) -> str:
        try:
            preferences = self.query_processor.extract_preferences(user_query)
            
            enriched_query = f"""
            Consulta del usuario: {user_query}
            
            Preferencias detectadas:
            - Géneros: {preferences['genres'] if preferences['genres'] else 'No especificado'}
            - Plataformas: {preferences['platforms'] if preferences['platforms'] else 'No especificado'}
            - Ordenamiento: {preferences['sort_preference'] if preferences['sort_preference'] else 'No especificado'}
            
            Por favor, usa las herramientas disponibles para encontrar los mejores juegos según estas preferencias.
            """
            
            print("*"*50)
            response = self.agent_executor.invoke({"input": enriched_query})
            print("/"*50)
            return response.get("output", str(response))
        except Exception as e:
            import traceback
            print(f"Error completo: {traceback.format_exc()}")
            return f"Error procesando la consulta: {str(e)}"

def main():
    print("🎮 Agente Recomendador de Juegos Gratuitos")
    print("=" * 50)
    
    agent = GameRecommendationAgent()
    
    while True:
        print("\n" + "=" * 50)
        user_input = input("\n¿Qué tipo de juegos te interesan? (o 'salir' para terminar): ")
        
        if user_input.lower() in ['salir', 'exit', 'quit']:
            print("¡Hasta luego! Que disfrutes jugando")
            break
        
        print("\nBuscando recomendaciones...")
        response = agent.recommend_games(user_input)
        print(f"\n{response}")

if __name__ == "__main__":
    main()
