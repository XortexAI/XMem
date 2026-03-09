from src.models import get_model

def test_bedrock_nova():
    print("Initializing Bedrock model from XMem registry...")
    
    # This will use the AWS credentials and bedrock settings from your .env
    # and return the ChatBedrockConverse instance we configured
    llm = get_model("bedrock")
    
    messages = [
        ("user", "Write a very short poem about memory.")
    ]
    
    print(f"Invoking model: {llm.model_id}")
    print("Waiting for response...\n")
    
    # Invoke the model using LangChain
    response = llm.invoke(messages)
    
    print("=== Model Response ===")
    print(response.content)
    print("======================")

if __name__ == "__main__":
    test_bedrock_nova()
