from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, TypeVar
from pydantic import BaseModel

ArgsT = TypeVar("ArgsT", bound=BaseModel)
ToolHandler = Callable[[ArgsT], Awaitable[dict[str, Any]]]

@dataclass(slots=True)
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    handler: ToolHandler

    def to_openai_tool(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args_model.model_json_schema(),
            }
        }

class ReadFileArgs(BaseModel):
    path: str

async def read_file(args: ReadFileArgs) -> dict[str, Any]:
    return {
        "path": args.path,
        "content": Path(args.path).read_text(encoding="utf-8"),
    }

READ_FILE_TOOL = Tool(
    name="read_file",
    description="Read a file and return the content",
    args_model=ReadFileArgs,
    handler=read_file,
)

