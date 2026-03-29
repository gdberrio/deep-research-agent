from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

endpoint = os.getenv("GPT-5-4-TARGET_URI")
deployment_name = "gpt-5.4"
api_key = os.getenv("GPT-5-4-API_KEY")

client = OpenAI(
    base_url=endpoint,
    api_key=api_key
)

completion = client.chat.completions.create(
    model=deployment_name,
    messages=[
        {
            "role": "user",
            "content": "What is the capital of France?",
        }
    ],
)

print(completion.choices[0].message)