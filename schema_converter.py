# schema_converter.py
import google.generativeai as genai
from google.ai.generativelanguage_v1beta.types import content
from typing import List, Dict, Callable
from tool_converters.tool_converter import ToolConverter

def map_json_type_to_content_type(json_type: str) -> content.Type:
    json_type = json_type.lower()
    if json_type == "string":
        return content.Type.STRING
    elif json_type == "number":
        return content.Type.NUMBER
    elif json_type == "boolean":
        return content.Type.BOOL
    elif json_type == "array":
        return content.Type.ARRAY
    elif json_type == "object":
        return content.Type.OBJECT
    else:
        return content.Type.OBJECT

def convert_schema_to_gemini(parameters: dict) -> content.Schema:
    schema_type = parameters.get("type", parameters.get("type_", "object"))
    ctype = map_json_type_to_content_type(schema_type)

    schema_builder = content.Schema(type=ctype)
    required_fields = parameters.get("required", [])
    if required_fields:
        schema_builder.required.extend(required_fields)

    if "enum" in parameters:
        schema_builder.enum[:] = parameters["enum"]

    props = parameters.get("properties", {})
    for prop_name, prop_schema in props.items():
        child_schema = convert_schema_to_gemini(prop_schema)
        schema_builder.properties[prop_name] = child_schema

    if ctype == content.Type.ARRAY:
        items = parameters.get("items", {})
        if items:
            item_schema = convert_schema_to_gemini(items)
            schema_builder.items.CopyFrom(item_schema)

    description = parameters.get("description")
    if description:
        schema_builder.description = description

    return schema_builder

def openai_to_gemini_tools(tools_schema: List[dict]) -> List[genai.protos.Tool]:
    declarations = []
    for tool in tools_schema:
        function_details = tool.get("function", tool)
        params = function_details.get("parameters", {})
        description = function_details.get("description", "")
        name = function_details.get("name", "unnamed_function")

        gemini_schema = convert_schema_to_gemini(params)
        fd = genai.protos.FunctionDeclaration(
            name=name,
            description=description,
            parameters=gemini_schema
        )
        declarations.append(fd)

    return [genai.protos.Tool(function_declarations=declarations)]

def convert_openai_to_anthropic(openai_schema: List[dict]) -> List[dict]:
    anthropic_schema = []
    for tool in openai_schema:
        function_details = tool.get("function", tool)
        anthropic_tool = {
            "name": function_details["name"],
            "description": function_details["description"],
            "input_schema": {
                "type": "object",
                "properties": function_details["parameters"]["properties"],
                "required": function_details["parameters"].get("required", [])
            }
        }
        anthropic_schema.append(anthropic_tool)
    return anthropic_schema

def generate_all_schemas(functions: List[Callable]) -> Dict[str, List[dict]]:
    """
    Generate schemas for all formats from a single list of Python functions.
    
    Returns:
    {
        "openai": [...],
        "anthropic": [...],
        "gemini": [...]
    }
    """
    converter = ToolConverter()
    # get OpenAI-like schema from the "gemini" key returned by ToolConverter
    full_schema = converter.generate_schemas(functions)
    openai_like_schema = []
    for entry in full_schema["gemini"]:
        if entry.get("type") == "function":
            openai_like_schema.append(entry)

    # Convert to Anthropic and Gemini
    anthropic_schema = convert_openai_to_anthropic(openai_like_schema)
    gemini_tools = openai_to_gemini_tools(openai_like_schema)

    return {
        "openai": openai_like_schema,
        "anthropic": anthropic_schema,
        "gemini": gemini_tools
    }