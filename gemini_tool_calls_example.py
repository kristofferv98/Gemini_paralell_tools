import os
import google.generativeai as genai
from google.ai.generativelanguage_v1beta.types import content

genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))

# Define Python functions for your tools
def subtract_numbers(a: float, b: float) -> float:
    print(f"Subtracting {a} and {b}")
    return a - b

def add_numbers(a: float, b: float) -> float:
    print(f"Adding {a} and {b}")
    return a + b

def multiply_numbers(a: float, b: float) -> float:
    print(f"Multiplying {a} and {b}")
    return a * b

def divide_numbers(a: float, b: float) -> float:
    print(f"Dividing {a} and {b}")
    return a / b

def square_number(a: float) -> float:
    print(f"Squaring {a}")
    return a ** 2

def cube_number(a: float) -> float:
    print(f"Cubing {a}")
    return a ** 3

# Map JSON schema types to Gemini content.Type
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

    # required fields
    required_fields = parameters.get("required", [])
    if required_fields:
        schema_builder.required.extend(required_fields)

    # enum
    if "enum" in parameters:
        schema_builder.enum[:] = parameters["enum"]

    # properties if object
    props = parameters.get("properties", {})
    for prop_name, prop_schema in props.items():
        child_schema = convert_schema_to_gemini(prop_schema)
        schema_builder.properties[prop_name] = child_schema

    # items if array
    if ctype == content.Type.ARRAY:
        items = parameters.get("items", {})
        if items:
            item_schema = convert_schema_to_gemini(items)
            schema_builder.items.CopyFrom(item_schema)

    # description
    description = parameters.get("description")
    if description:
        schema_builder.description = description

    return schema_builder

def convert_tool_schema_to_gemini(tools_schema: list[dict]) -> list[genai.protos.Tool]:
    declarations = []
    for tool in tools_schema:
        name = tool["name"]
        description = tool.get("description", "")
        params = tool.get("parameters", tool.get("input_schema", {}))
        gemini_schema = convert_schema_to_gemini(params)

        fd = genai.protos.FunctionDeclaration(
            name=name,
            description=description,
            parameters=gemini_schema
        )
        declarations.append(fd)

    return [genai.protos.Tool(function_declarations=declarations)]


class Assistant:
    def __init__(self, tools_schema: list[dict], functions_map: dict):
        """
        Initialize the assistant with a dynamic schema and a map of function names to callables.

        :param tools_schema: A list of tool definitions in a JSON-like schema (like OpenAI format)
        :param functions_map: A dict mapping function names (str) to Python callables
        """
        self.functions_map = functions_map

        # Convert the provided schema to Gemini format
        gemini_tools = convert_tool_schema_to_gemini(tools_schema)
        generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }

        self.model = genai.GenerativeModel(
            model_name="gemini-2.0-flash-exp",
            generation_config=generation_config,
            tools=gemini_tools,
        )
        print(self.model)

        self.history = []
        self.chat = self.model.start_chat(enable_automatic_function_calling=False, history=self.history)

    def send_user_message(self, user_text: str):
        self.history.append({"role": "user", "parts": [user_text]})
        return self.chat.send_message(user_text)

    def extract_function_calls(self, response) -> list[dict]:
        function_calls = []
        for candidate in response.candidates:
            for part in candidate.content.parts:
                if part.function_call:
                    function_calls.append({part.function_call.name: part.function_call.args})
        return function_calls

    def execute_functions(self, function_calls: list[dict]) -> list[dict]:
        results = []
        for fc in function_calls:
            fn_name = list(fc.keys())[0]
            fn_args = fc[fn_name]
            if fn_name in self.functions_map:
                result = self.functions_map[fn_name](**fn_args)
                results.append({fn_name: result})
                print(f"Function {fn_name} executed with result: {result}")
            else:
                results.append({fn_name: "Error: Function not found."})
        return results

    def send_function_response(self, function_results: list[dict]):
        response_parts = []
        for res in function_results:
            fn_name = list(res.keys())[0]
            value = res[fn_name]
            response_parts.append(
                genai.protos.Part(
                    function_response=genai.protos.FunctionResponse(
                        name=fn_name, 
                        response={"result": value}
                    )
                )
            )
        return self.chat.send_message(response_parts)

    def run_query(self, user_input: str):
        response = self.send_user_message(user_input)
        while True:
            function_calls = self.extract_function_calls(response)
            if not function_calls:
                break
            results = self.execute_functions(function_calls)
            response = self.send_function_response(results)

        final_text = ""
        for candidate in response.candidates:
            for part in candidate.content.parts:
                if part.text:
                    final_text += part.text
        return final_text.strip()


if __name__ == "__main__":
    # Example tools schema (like the openAI one with summarize_text and calculate_area)
    tools_schema_openai = [
        {
            "name": "add_numbers",
            "description": "Adds two numbers",
            "strict": True,
            "parameters": {
                "type": "object",
                "required": ["a", "b"],
                "properties": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"}
                },
                "additionalProperties": False
            }
        },
        {
            "name": "subtract_numbers", 
            "description": "Subtracts two numbers",
            "strict": True,
            "parameters": {
                "type": "object",
                "required": ["a", "b"],
                "properties": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"}
                },
                "additionalProperties": False
            }
        },
        {
            "name": "multiply_numbers",
            "description": "Multiplies two numbers",
            "strict": True,
            "parameters": {
                "type": "object",
                "required": ["a", "b"],
                "properties": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"}
                },
                "additionalProperties": False
            }
        },
        {
            "name": "divide_numbers",
            "description": "Divides two numbers",
            "strict": True,
            "parameters": {
                "type": "object",
                "required": ["a", "b"],
                "properties": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"}
                },
                "additionalProperties": False
            }
        },
        {
            "name": "square_number",
            "description": "Squares a number",
            "strict": True,
            "parameters": {
                "type": "object",
                "required": ["a"],
                "properties": {
                    "a": {"type": "number", "description": "Number to square"}
                },
                "additionalProperties": False
            }
        },
        {
            "name": "cube_number",
            "description": "Cubes a number",
            "strict": True,
            "parameters": {
                "type": "object",
                "required": ["a"],
                "properties": {
                    "a": {"type": "number", "description": "Number to cube"}
                },
                "additionalProperties": False
            }
        }
    ]

    # Map function names to actual callables
    functions_map = {
        "add_numbers": add_numbers,
        "subtract_numbers": subtract_numbers,
        "multiply_numbers": multiply_numbers,
        "divide_numbers": divide_numbers,
        "square_number": square_number,
        "cube_number": cube_number,
    }

    assistant = Assistant(tools_schema_openai, functions_map)

    # Test a query
    query = "Can you test the functions we have defined? And report the results as a detailed summary'?"
    answer = assistant.run_query(query)
    print("Final Answer:")
    print(answer)

