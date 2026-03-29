from openai import AsyncOpenAI
import json
from dotenv import load_dotenv
from rich import print
import os
from tools import READ_FILE_TOOL, Tool
import asyncio

load_dotenv()

endpoint = os.getenv("GPT-5-4-TARGET_URI")
deployment_name = "gpt-5.4"
api_key = os.getenv("GPT-5-4-API_KEY")

class AgentRuntime:
    def __init__(self, tools: list[Tool]) -> None:
        self.tools = {tool.name: tool for tool in tools}

    def get_tools(self) -> list[dict]:
        return [tool.to_openai_tool() for tool in self.tools.values()]

    async def execute_function_call(self, call: dict) -> dict:
        tool = self.tools[call.function.name]
        if tool is None:
            raise RuntimeError(f"Tool {call.function.name} not found")
        if call.function.arguments is None:
            raise RuntimeError(f"No arguments provided for tool {call.function.name}")
        args = tool.args_model.model_validate_json(call.function.arguments)
        result = await tool.handler(args)
        return {
            "role": "tool",
            "tool_call_id": call.id,
            "content": json.dumps(result)
        }
            
async def main() -> None:
    client = AsyncOpenAI(
        base_url=endpoint,
        api_key=api_key
    )
    runtime = AgentRuntime(tools=[READ_FILE_TOOL])

    input_list = [
        {
            "role": "user",
            "content": "Please read the README.md file and return the content.",
        }
    ]

    completion = await client.chat.completions.create(
        model=deployment_name,
        messages=input_list,
        tools=runtime.get_tools()
    )
    
    tool_calls = completion.choices[0].message.tool_calls or []
    function_calls = [x for x in tool_calls if x.type == "function"]
    if not function_calls:
        raise RuntimeError("No function calls found in completion")

    if len(function_calls) != 1:
        raise RuntimeError("Multiple function calls found in completion")

    function_call = function_calls[0]
    tool_result = await runtime.execute_function_call(function_call)

    input_list.append(completion.choices[0].message)
    input_list.append(tool_result)

    completion = await client.chat.completions.create(
        model=deployment_name,
        messages=input_list,
        tools=runtime.get_tools()
    )

    print(completion.choices[0].message.content)

if __name__ == "__main__":
    asyncio.run(main())
