# tool_converter_example.py
import inspect
import json
from typing import List, Callable, Dict
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor
import os

class ToolConverter:
    def __init__(self, indent_size: int = 4):
        self.indent_size = indent_size
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))
        self.model = genai.GenerativeModel("gemini-1.5-flash-latest")

    def _get_function_source(self, func: Callable) -> str:
        """Gets the complete source code of a function, including docstring."""
        source = inspect.getsource(func)
        # Remove any extra indentation from the source
        lines = source.split('\n')
        if lines:
            # Find the indentation of the first line
            first_line_indent = len(lines[0]) - len(lines[0].lstrip())
            # Remove that indentation from all lines
            lines = [line[first_line_indent:] if line.startswith(' ' * first_line_indent) else line 
                    for line in lines]
        return '\n'.join(lines)

    def convert_functions_to_string(self, functions: List[Callable]) -> List[str]:
        """
        Converts a list of Python functions into a list of strings, each containing a complete function definition.
        
        :param functions: List of Python functions
        :return: List of strings, each containing a function definition
        """
        function_strings = []
        for func in functions:
            function_strings.append(self._get_function_source(func))
        
        return function_strings
    
    def create_function_schema(self, function_string: str) -> Dict:
        """
        Creates a function schema from a string containing a function definition using Gemini.
        """
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
        
        # Extract the JSON from the response
        try:
            return json.loads(response.text)
        except json.JSONDecodeError:
            # If the response isn't valid JSON, try to extract it from the text
            text = response.text
            start = text.find('{')
            end = text.rfind('}') + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    return {"error": "Failed to parse Gemini response"}
            return {"error": "Failed to parse Gemini response"}
    
    def create_function_schemas(self, function_strings: List[str], max_workers: int = None) -> str:
        """
        Creates function schemas for multiple functions in parallel using ThreadPool
        and combines them into a single JSON string.
        
        :param function_strings: List of function definition strings
        :param max_workers: Maximum number of worker threads (defaults to None, which lets ThreadPoolExecutor decide)
        :return: Combined JSON string containing all function schemas
        """
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Map each function string to create_function_schema
            schema_futures = executor.map(self.create_function_schema, function_strings)
            
            # Collect all schemas
            schemas = []
            for schema in schema_futures:
                if "error" not in schema:
                    schemas.append(schema)
            
            # Combine all schemas into a single JSON string
            combined_schema = json.dumps(schemas, indent=2)
            return combined_schema

    def generate_schemas(self, functions: List[Callable]) -> dict:
        """
        Generates Gemini function schemas.
        
        Args:
            functions: List of Python functions to convert
            
        Returns:
            dict: Dictionary containing Gemini schema
            {
                "gemini": [...]
            }
        """
        # Convert functions to strings
        function_strings = self.convert_functions_to_string(functions)
        
        # Generate Gemini schema
        gemini_schema = self.create_function_schemas(function_strings)
        gemini_parsed = json.loads(gemini_schema)
        
        # Return Gemini schema
        return {
            "gemini": gemini_parsed
        }

# Example usage
def print_text(text: str) -> str:
    """Prints any text sent to the function and returns confirmation"""
    print(text)
    return "Text is printed"

def add_numbers(a: float, b: float) -> float:
    """Adds two numbers together"""
    return a + b

if __name__ == "__main__":
    # Create converter and generate schema
    converter = ToolConverter()
    functions = [print_text, add_numbers]
    schemas = converter.generate_schemas(functions)

    # Print Gemini schema
    print("Gemini Schema:")
    print(json.dumps(schemas["gemini"], indent=2))
