# tool_converter.py
import os
import json
import inspect
from typing import List, Callable, Dict
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor

class ToolConverter:
    """
    Converts Python functions into OpenAI-like function schemas using a generative model.
    """

    def __init__(self, model_name: str = "gemini-2.0-flash-exp", indent_size: int = 4):
        self.indent_size = indent_size
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))
        self.model = genai.GenerativeModel(model_name)

    def _get_function_source(self, func: Callable) -> str:
        source = inspect.getsource(func)
        lines = source.split('\n')
        if lines:
            first_line_indent = len(lines[0]) - len(lines[0].lstrip())
            lines = [line[first_line_indent:] if line.startswith(' ' * first_line_indent) else line
                     for line in lines]
        return '\n'.join(lines)

    def convert_functions_to_string(self, functions: List[Callable]) -> List[str]:
        return [self._get_function_source(func) for func in functions]
    
    def create_function_schema(self, function_string: str) -> Dict:
        prompt = f"""Generate a JSON schema for the following Python function. The schema should follow this format:
        {{
            "type": "function",
            "function": {{
                "name": "function_name",
                "description": "function description",
                "strict": true,
                "parameters": {{
                    "type": "object",
                    "properties": {{
                        // parameter definitions
                    }},
                    "required": ["param1", "param2", ...],
                    "additionalProperties": false
                }}
            }}
        }}

        Function to convert:
        {function_string}
        """

        response = self.model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.2,
                top_p=0.95,
                top_k=40,
                max_output_tokens=8192,
                response_mime_type="application/json"
            )
        )
        
        text = response.text
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find('{')
            end = text.rfind('}') + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    return {"error": "Failed to parse Gemini response"}
            return {"error": "Failed to parse Gemini response"}
    
    def create_function_schemas(self, function_strings: List[str], max_workers: int = None) -> str:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            schema_results = executor.map(self.create_function_schema, function_strings)

        schemas = [res for res in schema_results if "error" not in res]
        return json.dumps(schemas, indent=self.indent_size)

    def generate_schemas(self, functions: List[Callable]) -> dict:
        function_strings = self.convert_functions_to_string(functions)
        gemini_schema_str = self.create_function_schemas(function_strings)
        gemini_parsed = json.loads(gemini_schema_str)
        # Returns a dict with "gemini" key containing OpenAI-like schemas
        return {"gemini": gemini_parsed}
    