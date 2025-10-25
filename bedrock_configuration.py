import os
from strands import Agent
from strands.models import BedrockModel

# Configurar la API Key de Bedrock como variable de entorno
# IMPORTANTE: Reemplaza 'TU_API_KEY_DE_BEDROCK' con tu clave real
os.environ['AWS_BEARER_TOKEN_BEDROCK'] = 'TU_API_KEY_DE_BEDROCK'

def main():
    """
    Función principal que configura el agente y responde una pregunta
    """

    print("Configurando el agente con Amazon Nova Lite...\n")

    # Configurar el modelo Amazon Nova Lite 1.0
    # Nova Lite es un modelo multimodal rápido y de bajo costo
    nova_lite_model = BedrockModel(
        model_id="amazon.nova-lite-v1:0",  # ID del modelo Nova Lite
        region_name="us-east-1",               # Región de AWS
        temperature=0.7,                       # Controla la creatividad (0.0-1.0)
    )

    # Crear el agente con el modelo configurado
    agent = Agent(
        model=nova_lite_model,
        system_prompt="Eres un asistente útil y conocedor sobre inteligencia artificial y agentes."
    )

    # Hacer la pregunta
    pregunta = "¿Qué es un agente?"

    print(f"Pregunta: {pregunta}\n")
    print("Generando respuesta...")
    print("=" * 60)

    # Obtener la respuesta del agente
    respuesta = agent(pregunta)

    # Mostrar la respuesta
    print(respuesta)
    print("=" * 60)
    print("\n✅ Respuesta generada exitosamente!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n⚠️  Asegúrate de que:")
        print("   1. Has reemplazado 'TU_API_KEY_DE_BEDROCK' con tu clave real")
        print("   2. Tu API Key es válida y no ha expirado")
        print("   3. Tienes acceso habilitado al modelo Nova Lite en Bedrock")
        print("   4. La región us-east-1 está disponible para tu cuenta")
        print("\n💡 Para generar una API Key:")
        print("   - Ve a la consola de AWS Bedrock")
        print("   - Selecciona 'API keys' en el menú lateral")
        print("   - Genera una clave de corto plazo (12 horas) o largo plazo (30 días)")