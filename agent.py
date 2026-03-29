from openai import OpenAI
import json
from dotenv import load_dotenv
from rich import print
import os
from pathlib import Path

load_dotenv()

endpoint = os.getenv("GPT-5-4-TARGET_URI")
deployment_name = "gpt-5.4"
api_key = os.getenv("GPT-5-4-API_KEY")

def read_file(file_path: str) -> str:
    return Path(file_path).read_text()

tools = [
    {
        "type": "function",
            "name": "read_file",
            "description": "Read a file and return the content",
            "parameters": {
                "type": "object",
                "properties": {"file_path": {"type": "string", "description": "The path to the file to read"}},
                "required": ["file_path"]
            }
        }
]

client = OpenAI(
    base_url=endpoint,
    api_key=api_key
)

input_list = [
    {
        "role": "user",
        "content": "Please read the README.md file and return the content.",
    }
]

completion = client.responses.create(
    model=deployment_name,
    input=input_list,
    tools=tools
)

input_list += completion.output
print(completion.output)

for item in completion.output:
    if item.type == "function_call":
        if item.name == "read_file":
            file_path = json.loads(item.arguments)["file_path"]
            file_content = read_file(file_path)
            input_list.append({
                "type": "function_call_output",
                "call_id": item.call_id,
                "output": file_content
            })

print("final input list:")
print(input_list)

response = client.responses.create(
    model=deployment_name,
    input=input_list,
    tools=tools
)

print("final response:")
print(response.model_dump_json(indent=2))
print(response.output_text)