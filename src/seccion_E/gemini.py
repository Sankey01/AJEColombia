"""
Cliente para interactuar con la API de Gemini 2.5 Flash usando google-genai
"""
import os
from google import genai
from google.genai import types
from typing import Optional, Dict, Any, Generator, List
import json
import time
from .config import Config
from .logger_config import get_module_logger

logger = get_module_logger("gemini_client")

class GeminiClient:
    """Cliente para la API de Gemini 2.5 Flash"""

    def __init__(self):
        """Inicializa el cliente de Gemini 2.5 Flash"""
        try:
            Config.validate_config()

            self.client = genai.Client(api_key=Config.GEMINI_API_KEY)
            self.model = Config.GEMINI_MODEL

            logger.info(f"Cliente Gemini 2.5 Flash inicializado correctamente")
            logger.info(f"Modelo: {self.model}")
            logger.info(f"Max tokens: {Config.GEMINI_MAX_TOKENS}")
            logger.info(f"Temperature: {Config.GEMINI_TEMPERATURE}")

        except Exception as e:
            logger.error(f"Error al inicializar cliente Gemini: {str(e)}")
            raise

    def generate_content(self, prompt: str, use_google_search: bool = False, thinking_budget: Optional[int] = None) -> Dict[str, Any]:
        """
        Genera contenido usando Gemini 2.5 Flash (sin streaming)

        Args:
            prompt (str): El prompt a enviar a Gemini
            use_google_search (bool): Si usar Google Search
            thinking_budget (int): Budget para thinking (None usa config)

        Returns:
            Dict[str, Any]: Respuesta con el contenido generado y metadatos
        """
        try:
            logger.info(f"Enviando prompt a Gemini (longitud: {len(prompt)} caracteres)")

            start_time = time.time()

            # Preparar contenido
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt)]
                )
            ]

            # Configurar herramientas
            tools = []
            if use_google_search:
                tools.append(types.Tool(googleSearch=types.GoogleSearch()))
                logger.info("Google Search habilitado")

            # Configurar generación
            thinking_budget_val = thinking_budget if thinking_budget is not None else Config.GEMINI_THINKING_BUDGET

            config = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(
                    thinking_budget=thinking_budget_val
                ),
                tools=tools if tools else None
            )

            # Generar contenido (sin streaming)
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config
            )

            end_time = time.time()
            response_time = end_time - start_time

            # Extraer texto de la respuesta
            content_text = ""
            if response.candidates:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'text'):
                        content_text += part.text

            result = {
                "success": True,
                "content": content_text,
                "response_time": response_time,
                "model": self.model,
                "prompt_length": len(prompt),
                "response_length": len(content_text),
                "thinking_budget": thinking_budget_val,
                "google_search_used": use_google_search,
                "timestamp": time.time(),
                "raw_response": response
            }

            logger.info(f"Respuesta recibida exitosamente (tiempo: {response_time:.2f}s, "
                       f"longitud: {result['response_length']} caracteres)")

            return result

        except Exception as e:
            logger.error(f"Error al generar contenido: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "content": None,
                "timestamp": time.time()
            }

    def generate_content_stream(self, prompt: str, use_google_search: bool = False, thinking_budget: Optional[int] = None) -> Generator[Dict[str, Any], None, None]:
        """
        Genera contenido usando streaming de Gemini 2.5 Flash

        Args:
            prompt (str): El prompt a enviar a Gemini
            use_google_search (bool): Si usar Google Search
            thinking_budget (int): Budget para thinking

        Yields:
            Dict[str, Any]: Chunks de respuesta con metadatos
        """
        try:
            logger.info(f"Iniciando stream a Gemini (longitud: {len(prompt)} caracteres)")

            start_time = time.time()
            full_content = ""
            chunk_count = 0

            # Preparar contenido
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt)]
                )
            ]

            # Configurar herramientas
            tools = []
            if use_google_search:
                tools.append(types.Tool(googleSearch=types.GoogleSearch()))
                logger.info("Google Search habilitado para streaming")

            # Configurar generación
            thinking_budget_val = thinking_budget if thinking_budget is not None else Config.GEMINI_THINKING_BUDGET

            config = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(
                    thinking_budget=thinking_budget_val
                ),
                tools=tools if tools else None
            )

            # Stream de contenido
            for chunk in self.client.models.generate_content_stream(
                model=self.model,
                contents=contents,
                config=config
            ):
                chunk_count += 1
                chunk_text = chunk.text if hasattr(chunk, 'text') else ""
                full_content += chunk_text

                yield {
                    "success": True,
                    "chunk_text": chunk_text,
                    "chunk_number": chunk_count,
                    "full_content_so_far": full_content,
                    "is_complete": False,
                    "timestamp": time.time()
                }

            end_time = time.time()
            response_time = end_time - start_time

            # Chunk final con estadísticas completas
            yield {
                "success": True,
                "chunk_text": "",
                "chunk_number": chunk_count,
                "full_content_so_far": full_content,
                "is_complete": True,
                "response_time": response_time,
                "total_chunks": chunk_count,
                "total_length": len(full_content),
                "model": self.model,
                "thinking_budget": thinking_budget_val,
                "google_search_used": use_google_search,
                "timestamp": time.time()
            }

            logger.info(f"Stream completado (tiempo: {response_time:.2f}s, "
                       f"chunks: {chunk_count}, longitud: {len(full_content)})")

        except Exception as e:
            logger.error(f"Error en streaming: {str(e)}")
            yield {
                "success": False,
                "error": str(e),
                "is_complete": True,
                "timestamp": time.time()
            }

    def generate_with_conversation(self, messages: List[Dict[str, str]], use_google_search: bool = False) -> Dict[str, Any]:
        """
        Genera contenido con historial de conversación

        Args:
            messages (List[Dict]): Lista de mensajes [{"role": "user/model", "content": "texto"}]
            use_google_search (bool): Si usar Google Search

        Returns:
            Dict[str, Any]: Respuesta con el contenido generado
        """
        try:
            logger.info(f"Generando con conversación ({len(messages)} mensajes)")

            contents = []
            for msg in messages:
                role = "user" if msg["role"] == "user" else "model"
                contents.append(
                    types.Content(
                        role=role,
                        parts=[types.Part.from_text(text=msg["content"])]
                    )
                )

            tools = []
            if use_google_search:
                tools.append(types.Tool(googleSearch=types.GoogleSearch()))

            config = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(
                    thinking_budget=Config.GEMINI_THINKING_BUDGET
                ),
                tools=tools if tools else None
            )

            start_time = time.time()

            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config
            )

            end_time = time.time()

            content_text = ""
            if response.candidates:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'text'):
                        content_text += part.text

            result = {
                "success": True,
                "content": content_text,
                "response_time": end_time - start_time,
                "conversation_length": len(messages),
                "google_search_used": use_google_search,
                "timestamp": time.time()
            }

            logger.info("Conversación procesada exitosamente")
            return result

        except Exception as e:
            logger.error(f"Error en conversación: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "content": None,
                "timestamp": time.time()
            }

    def test_connection(self) -> Dict[str, Any]:
        """Prueba la conexión con la API de Gemini 2.5 Flash"""
        try:
            logger.info("Probando conexión con API de Gemini 2.5 Flash")

            test_prompt = "Responde solo con 'Conexión exitosa con Gemini 2.5 Flash' si puedes leer este mensaje."
            result = self.generate_content(test_prompt)

            if result["success"]:
                logger.info("Prueba de conexión exitosa")
                return {
                    "success": True,
                    "message": "Conexión con Gemini 2.5 Flash API exitosa",
                    "model": self.model,
                    "response": result["content"],
                    "response_time": result["response_time"]
                }
            else:
                logger.error("Prueba de conexión fallida")
                return {
                    "success": False,
                    "message": "Error en la conexión con Gemini 2.5 Flash API",
                    "error": result.get("error")
                }

        except Exception as e:
            logger.error(f"Error durante la prueba de conexión: {str(e)}")
            return {
                "success": False,
                "message": "Error durante la prueba de conexión",
                "error": str(e)
            }

    def get_model_info(self) -> Dict[str, Any]:
        """Obtiene información del modelo Gemini 2.5 Flash configurado"""
        return {
            "model_name": self.model,
            "temperature": Config.GEMINI_TEMPERATURE,
            "max_tokens": Config.GEMINI_MAX_TOKENS,
            "thinking_budget": Config.GEMINI_THINKING_BUDGET,
            "api_configured": bool(Config.GEMINI_API_KEY),
            "version": "Gemini 2.5 Flash (gratuito)"
        }