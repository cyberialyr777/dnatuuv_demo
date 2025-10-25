import os
from dotenv import load_dotenv
from strands import Agent
from strands.models import BedrockModel


load_dotenv()
AWS_BEARER_TOKEN_BEDROCK = os.getenv("AWS_BEARER_TOKEN_BEDROCK")

def main():

    nova_lite_model = BedrockModel(
        model_id="amazon.nova-lite-v1:0", 
        region_name="us-east-1",              
        temperature=0.7,                       
    )
    
    agent = Agent(
        model=nova_lite_model,
        system_prompt="Eres un asistente útil y conocedor sobre inteligencia artificial y agentes."
    )
    
    pregunta = "¿Qué es un agente?"

    print(f"Pregunta: {pregunta}\n")
    print("Generando respuesta...")
    print("=" * 60)

    respuesta = agent(pregunta)

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