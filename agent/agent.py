from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageToolCall
import json
from rich import print
from tools import READ_FILE_TOOL, MODIFY_TODO_TOOL, Tool
import asyncio
from state import RunConfig, RunState, AgentContext


class AgentRuntime:
    def __init__(self, config: RunConfig, state: RunState, context: AgentContext, tools: list[Tool]) -> None:
        self.config = config
        self.state = state
        self.context = context
        self.tools = {tool.name: tool for tool in tools}

    def get_tools(self) -> list[dict]:
        if self.state.iteration_count >= self.config.max_iterations:
            return []
        return [tool.to_openai_tool() for tool in self.tools.values()]

    async def execute_function_call(self, call: ChatCompletionMessageToolCall) -> dict:
        tool = self.tools.get(call.function.name)
        if tool is None:
            raise RuntimeError(f"Tool {call.function.name} not found")
        if call.function.arguments is None:
            raise RuntimeError(f"No arguments provided for tool {call.function.name}")
        args = tool.args_model.model_validate_json(call.function.arguments)
        result = await tool.handler(args, self.state, self.context)
        return {
            "role": "tool",
            "tool_call_id": call.id,
            "content": json.dumps(result)
        }
            
async def main() -> None:
    config = RunConfig(max_iterations=5)
    state = RunState()
    context = AgentContext()
    client = AsyncOpenAI(
        base_url=config.model_target_uri,
        api_key=config.model_api_key
    )
    runtime = AgentRuntime(config, state, context, [READ_FILE_TOOL, MODIFY_TODO_TOOL])

    input_list = [
        {
            "role": "user",
            "content": "First add a todo to read README.md, then read the README.md file.",
        }
    ]

    while True:
        state.iteration_count += 1
        print(
            {
                "iteration_count": state.iteration_count,
                "max_iterations": config.max_iterations,
                "todos": state.todos,
            }
        )

        completion = await client.chat.completions.create(
            model=config.model,
            messages=input_list,
            tools=runtime.get_tools()
        )
        
        tool_calls = completion.choices[0].message.tool_calls or []
        function_calls = [x for x in tool_calls if x.type == "function"]
        if not function_calls:
            print("Ending loop: No more function calls found in completion")
            break

        for function_call in function_calls:
            print(function_call)
            tool_result = await runtime.execute_function_call(function_call)
            if tool_result.get("error") is not None:
                print(f"Error executing function call {function_call.function.name}: {tool_result['error']}")
                continue
            input_list.append(completion.choices[0].message)
            input_list.append(tool_result)

    print(input_list)

if __name__ == "__main__":
    asyncio.run(main())
