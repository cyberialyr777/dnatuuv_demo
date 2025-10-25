# -*- coding: utf-8 -*-

"""
Este script implementa un bot de Telegram que actúa como un asistente de bienestar (Aura)
utilizando un modelo de lenguaje de AWS Bedrock (amazon.nova-lite-v1:0).

Arquitectura clave:
1.  Usa `python-telegram-bot` para manejar la API de Telegram (recibir y enviar mensajes).
2.  Usa una biblioteca personalizada `strands` que abstrae la lógica del LLM.
3.  Mantiene un estado de conversación (historial) único para cada usuario utilizando
    `context.user_data` de `python-telegram-bot`.
"""

import os
from dotenv import load_dotenv
import logging
from strands import Agent
from strands.models import BedrockModel
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

load_dotenv()
AWS_BEARER_TOKEN_BEDROCK = os.getenv("AWS_BEARER_TOKEN_BEDROCK")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") 

# Configura el sistema de logging. Esto es crucial para depurar en producción.
# Nos permite ver qué está pasando, quién está hablando con el bot y qué errores ocurren.
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- Configuración del Modelo de Lenguaje (LLM) ---

# Esta es la configuración base para el modelo que se usará al crear nuevos agentes.
# NOTA: `BedrockModel` es una clase de la librería `strands` que actúa como
# cliente para la API de AWS Bedrock.
bedrock_model = BedrockModel(
    model_id="amazon.nova-lite-v1:0",  # El modelo específico de Bedrock a usar
    region_name="us-east-1",          # Región de AWS donde está el modelo
    temperature=0.7,
)

# --- El "Cerebro" del Agente (System Prompt) ---
# Esta es la instrucción central que define la personalidad, conocimiento y lógica del bot.
# Es la parte más importante de la "Ingeniería de Prompts".
DNATUUV_SYSTEM_PROMPT = """
# System Prompt - Asistente dnatuuv

. Definición y Persona del Agente

Nombre: Aura
... (El resto del prompt que define el catálogo, lógica de diagnóstico, reglas, etc.) ...
"""

# --- Funciones de Gestión de Agentes (Gestión de Estado) ---

def create_new_agent() -> Agent:
    """
    Función "Factory" (fábrica) para crear una NUEVA instancia de un agente.
    Cada agente es independiente y comienza con un historial limpio,
    pero comparte la misma configuración de modelo y system prompt.
    """
    # Crea una nueva instancia del modelo. Esto podría ser importante si la
    # clase `BedrockModel` mantiene algún estado interno, aunque aquí parece
    # más una re-instanciación para asegurar un estado limpio.
    fresh_bedrock_model = BedrockModel(
        model_id="amazon.nova-lite-v1:0",
        region_name="us-east-1",
        temperature=0.7,
    )
    
    # Crea la instancia del Agente
    dnatuuv_agent = Agent(
        model=fresh_bedrock_model,
        system_prompt=DNATUUV_SYSTEM_PROMPT,
    )
    logger.info("Nueva instancia de Agent creada.")
    return dnatuuv_agent

def get_agent_for_user(context: ContextTypes.DEFAULT_TYPE) -> Agent:
    """
    Gestiona el estado de la conversación por usuario.
    Recupera el agente existente de un usuario desde `context.user_data` o
    crea uno nuevo si es la primera vez que el usuario interactúa.

    Esto es CRUCIAL para que el bot pueda "recordar" la conversación
    con cada usuario de forma independiente.
    """
    if 'agent' not in context.user_data:
        logger.info(f"Creando nuevo agente para el usuario {context._user_id}")
        context.user_data['agent'] = create_new_agent()
    return context.user_data['agent']

# --- Manejadores de Comandos de Telegram (Handlers) ---

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Manejador para el comando /start. Se activa cuando un usuario
    inicia la conversación por primera vez.
    """
    logger.info(f"Usuario {update.effective_user.id} inició el bot con /start.")
    agent = get_agent_for_user(context)
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action=ChatAction.TYPING
    )
    
    welcome_message = agent(
        "Saluda al usuario de forma cálida y breve. Preséntate y pregunta cómo puedes ayudarle hoy. Mantén tu saludo en máximo 3 líneas."
    )

    commands_info = "\n\nPD: Si en algún momento quieres empezar de cero o reiniciar nuestra conversación, solo escribe: \n/nuevo\n/reset"
    
    await update.message.reply_text(f"🌿 Aura: {welcome_message}{commands_info}")

async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Manejador para los comandos /nuevo o /reset.
    Permite al usuario borrar el historial de conversación y empezar de cero.
    """
    logger.info(f"Usuario {update.effective_user.id} reinició la conversación.")
    context.user_data['agent'] = create_new_agent()
    agent = context.user_data['agent'] 

    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action=ChatAction.TYPING
    )

    welcome_message = agent(
        "Saluda nuevamente al usuario de forma breve y pregunta cómo puedes ayudarle."
    )
    
    await update.message.reply_text(f"✨ Conversación reiniciada.\n\n🌿 Aura: {welcome_message}")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Manejador principal para todos los mensajes de texto que NO son comandos.
    Este es el bucle principal de la conversación.
    """
    user_input = update.message.text
    logger.info(f"Mensaje de {update.effective_user.id}: {user_input}")

    # 1. Obtiene el agente EXISTENTE del usuario (con su historial).
    agent = get_agent_for_user(context)
    
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action=ChatAction.TYPING
    )
    
    try:
        # 2. Pasa el input del usuario al agente.
        # La librería `strands` (internamente):
        #    a. Añade `user_input` al historial de chat.
        #    b. Construye un prompt completo (System Prompt + Historial + Nuevo Input).
        #    c. Llama a la API de AWS Bedrock.
        #    d. Recibe la respuesta del LLM.
        #    e. Añade la respuesta del LLM al historial.
        #    f. Devuelve la respuesta de texto.
        response = agent(user_input)
        
        # 3. Envía la respuesta generada al usuario.
        await update.message.reply_text(f"🌿 Aura: {response}")
        
    except Exception as e:
        logger.error(f"Error procesando mensaje para {update.effective_user.id}: {e}", exc_info=True)
        await update.message.reply_text("Lo siento, ocurrió un error inesperado. Por favor, intenta de nuevo. Si el problema persiste, puedes escribir /nuevo para reiniciar nuestra conversación.")

def main() -> None:
    """
    Función principal que configura e inicia el bot.
    """
    if not TELEGRAM_BOT_TOKEN:
        logger.error("¡ERROR! No se encontró el TELEGRAM_BOT_TOKEN. Asegúrate de configurarlo en tu archivo .env")
        return

    logger.info("Iniciando bot...")
    
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start_command))

    application.add_handler(CommandHandler(["nuevo", "reset"], reset_command))

    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("El bot está corriendo. Presiona Ctrl+C para detenerlo.")

    application.run_polling()


if __name__ == "__main__":

    main()
