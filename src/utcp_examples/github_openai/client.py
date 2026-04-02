import asyncio
import os
from pathlib import Path
import json
import re

from dotenv import load_dotenv
from agents import Agent, Runner, FunctionTool

from utcp.utcp_client import UtcpClient
from utcp.data.utcp_client_config import UtcpClientConfigSerializer
from utcp.data.tool import Tool
from utcp_http.openapi_converter import OpenApiConverter
import aiohttp

# Convert any OpenAPI spec to UTCP tools
async def convert_api():
    async with aiohttp.ClientSession() as session:
        async with session.get("https://raw.githubusercontent.com/github/rest-api-description/main/descriptions/api.github.com/api.github.com.json") as response:
            openapi_spec = await response.json()
    
    converter = OpenApiConverter(openapi_spec)
    manual = converter.convert()
    
    print(f"Generated {len(manual.tools)} tools from GitHub API!")
    return manual

async def initialize_utcp_client() -> UtcpClient:
    """Initialize the UTCP client with configuration."""
    # Create a configuration for the UTCP client

    config = {
    "manual_call_templates": [{
        "name": "github",
        "call_template_type": "http", 
        "url": "https://raw.githubusercontent.com/github/rest-api-description/main/descriptions/api.github.com/api.github.com.json",
        "auth_tools": {  # Authentication for generated tools requiring auth
            "auth_type": "api_key",
            "api_key": "Bearer ${GITHUB_TOKEN}",
            "var_name": "Authorization",
            "location": "header"
        }
    }]
    }
    
    # Create and return the UTCP client
    return await UtcpClient.create(
        config=config
    )

def inline_refs(schema: dict, root: dict | None = None, _resolving: set | None = None) -> dict:
    """Recursively inline all $ref references so no nested refs remain."""
    if root is None:
        root = schema
    if _resolving is None:
        _resolving = set()
    if not isinstance(schema, dict):
        return schema

    if "$ref" in schema:
        ref = schema["$ref"]
        if not ref.startswith("#/"):
            return {}
        if ref in _resolving:
            return {"type": "object"}
        parts = ref.lstrip("#/").split("/")
        target = root
        for part in parts:
            target = target.get(part, {})
        _resolving = _resolving | {ref}
        return inline_refs(dict(target), root, _resolving)

    return {
        k: (
            inline_refs(v, root, _resolving) if isinstance(v, dict)
            else [inline_refs(i, root, _resolving) if isinstance(i, dict) else i for i in v]
            if isinstance(v, list)
            else v
        )
        for k, v in schema.items()
        if k not in ("$defs", "definitions")
    }

def ensure_type(schema: dict) -> dict:
    """Recursively ensure every schema object has a 'type' key, as required by OpenAI strict mode."""
    if not isinstance(schema, dict):
        return schema
    if "properties" in schema and isinstance(schema["properties"], dict):
        for key, val in schema["properties"].items():
            if not isinstance(val, dict):
                schema["properties"][key] = {"type": "string"}
    if "type" not in schema and "$ref" not in schema:
        if "properties" in schema:
            schema["type"] = "object"
        elif "items" in schema:
            schema["type"] = "array"
        elif not any(k in schema for k in ("oneOf", "anyOf", "allOf")):
            schema["type"] = "string"
    for value in schema.values():
        if isinstance(value, dict):
            ensure_type(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    ensure_type(item)
    return schema

def remove_additional_properties(schema: dict) -> dict:
    """Recursively remove additionalProperties from a JSON schema for OpenAI strict mode."""
    if not isinstance(schema, dict):
        return schema
    schema.pop("additionalProperties", None)
    for value in schema.values():
        if isinstance(value, dict):
            remove_additional_properties(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    remove_additional_properties(item)
    return schema

def sanitize_tool_name(name: str) -> str:
    """
    Sanitize tool name to match OpenAI's pattern requirement: ^[a-zA-Z0-9_-]+$
    """
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
    if not sanitized or not re.match(r'^[a-zA-Z0-9]', sanitized):
        sanitized = 'tool_' + sanitized
    return sanitized

def utcp_tool_to_agent_tool(utcp_client: UtcpClient, tool: Tool) -> FunctionTool:
    """
    Creates a FunctionTool that wraps a UTCP tool,
    making it compatible with the openai-agents library.
    """
    
    async def tool_invoke_handler(ctx, args: str) -> str:
        """
        Handler function for the UTCP tool invocation.
        """
        print(f"\n🤖 Agent is calling tool: {tool.name} with args: {args}")
        try:
            kwargs = json.loads(args) if args.strip() else {}
            
            result = await utcp_client.call_tool(tool.name, kwargs)
            print(f"✅ Tool {tool.name} executed successfully. Result: {result}")
            
            if isinstance(result, (dict, list)):
                return json.dumps(result)
            else:
                return str(result)
        except Exception as e:
            print(f"❌ Error calling tool {tool.name}: {e}")
            return f"Error: {str(e)}"

    params_schema = {"type": "object", "properties": {}, "required": []}
    
    if tool.inputs and tool.inputs.properties:
        inputs_dict = tool.inputs.model_dump(exclude_none=True)
        for prop_name, prop_schema in inputs_dict["properties"].items():
            if not isinstance(prop_schema, dict):
                prop_schema = {"type": "string"}
            params_schema["properties"][prop_name] = prop_schema
        
        if inputs_dict.get("required"):
            params_schema["required"] = inputs_dict["required"]

    sanitized_name = sanitize_tool_name(tool.name)
    try:
        return FunctionTool(
            name=sanitized_name,
            description=tool.description or f"No description available for {tool.name}.",
            params_json_schema=ensure_type(remove_additional_properties(inline_refs(params_schema))),
            on_invoke_tool=tool_invoke_handler,
        )
    except Exception:
        return None

async def main():
    """Main function to set up and run the OpenAI agent."""
    load_dotenv(Path(__file__).parent / ".env")
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: OPENAI_API_KEY not found. Please set it in your .env file.")
        return

    print("🚀 Initializing UTCP client...")
    try:
        utcp_client = await initialize_utcp_client()
        utcp_tools = await utcp_client.config.tool_repository.get_tools()
        print(f"✅ UTCP client initialized. Found {len(utcp_tools)} tools.")
    except Exception as e:
        print(f"❌ Failed to initialize UTCP client or fetch tools: {e}")
        print("   Is the server running? Try: uvicorn server:app --port 8080")
        return

    agent_tools = [t for t in (utcp_tool_to_agent_tool(utcp_client, tool) for tool in utcp_tools) if t is not None]
    print(f"✅ Converted {len(agent_tools)}/{len(utcp_tools)} tools for the agent.")

    gymbro_agent = Agent(
        name="GitHub Assistant",
        instructions="You are a helpful GitHub assistant with access to the GitHub API. Help the user with their Github inquiries.",
        model="gpt-4o-mini",
        tools=agent_tools,
    )

    print("\n--- GitHub Assistant is ready! ---")
    print("Type your request or 'exit' to quit.")

    while True:
        user_input = input("> ")
        if user_input.lower() == "exit":
            print("💪 Keep pushing! Goodbye!")
            break
        
        try:
            response_stream = await Runner.run(
                gymbro_agent,
                user_input
            )
            
            print(response_stream)
            
        except Exception as e:
            print(f"\nAn error occurred: {e}")


if __name__ == "__main__":
    asyncio.run(main())