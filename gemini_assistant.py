# gemini_assistant.py
import os
import google.generativeai as genai
from typing import List, Callable, Dict, Optional, Any
from tool_converters.tool_converter import ToolConverter
from tool_converters.schema_converter import openai_to_gemini_tools

genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))

class GeminiAssistant:
    """
    A Gemini-based assistant that can:
    - Accept a functions_map, a list of functions, or a pre-defined OpenAI tools_schema.
    - If given functions, it generates an OpenAI-like schema (referred as "gemini" schema) using ToolConverter.
    - Converts that schema to Gemini tools using schema_converter.
    - Initializes a Gemini model that can call these functions in parallel.
    - Provides run_query(...) to interact with the model: the model may request function calls, which the assistant executes.
    - Provides get_schema() to retrieve the generated OpenAI-like schema.

    Args:
        functions_map: A dict mapping function name to a callable. 
        functions_list: A list of Python callables. If provided, will auto-generate a functions_map.
        tools_schema: A schema in OpenAI format. If provided, no generation is done.
        model_name: The Gemini model name (default "gemini-2.0-flash-exp").
        generation_config: Dict of generation parameters.

    Raises:
        ValueError: If neither functions_map, functions_list nor tools_schema is provided.
    """

    def __init__(self,
                 functions_map: Optional[Dict[str, Callable]] = None,
                 functions_list: Optional[List[Callable]] = None,
                 tools_schema: Optional[List[dict]] = None,
                 model_name: str = "gemini-2.0-flash-exp",
                 generation_config: Optional[Dict[str, Any]] = None):
        if functions_map is None and functions_list is None and tools_schema is None:
            raise ValueError("Must provide either functions_map, functions_list, or tools_schema.")

        if functions_map is None and functions_list is not None:
            # Derive a function map from function list using their __name__
            functions_map = {f.__name__: f for f in functions_list}

        self.functions_map = functions_map

        if tools_schema is None:
            # Generate from provided functions using ToolConverter
            converter = ToolConverter()
            funcs = list(self.functions_map.values())
            full_schema = converter.generate_schemas(funcs)

            # full_schema["gemini"] is the OpenAI-like schema
            openai_like_schema = []
            for entry in full_schema["gemini"]:
                if entry.get("type") == "function":
                    openai_like_schema.append(entry)
            self.openai_schema = openai_like_schema
        else:
            self.openai_schema = tools_schema

        # Convert OpenAI-like schema to Gemini tools format
        self.gemini_tools = openai_to_gemini_tools(self.openai_schema)

        # Default generation config if not provided
        if generation_config is None:
            generation_config = {
                "temperature": 1,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 8192,
                "response_mime_type": "text/plain",
            }

        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config,
            tools=self.gemini_tools,
        )

        self.history = []
        self.chat = self.model.start_chat(enable_automatic_function_calling=False, history=self.history)

    def get_schema(self) -> List[dict]:
        """
        Return the generated OpenAI-like schema from the Python functions or the provided tools_schema.

        Returns:
            A list of dictionaries representing the OpenAI-like schema.
        """
        return self.openai_schema

    def send_user_message(self, user_text: str):
        """
        Send a user message to the model and update conversation history.

        Args:
            user_text: The user's query or prompt.

        Returns:
            The model's response object.
        """
        self.history.append({"role": "user", "parts": [user_text]})
        return self.chat.send_message(user_text)

    def extract_function_calls(self, response) -> List[dict]:
        """
        Extract any function calls requested by the model from the response.

        Args:
            response: The model response object.

        Returns:
            A list of dicts in the form [{function_name: {args}}].
        """
        function_calls = []
        for candidate in response.candidates:
            for part in candidate.content.parts:
                if part.function_call:
                    function_calls.append({part.function_call.name: part.function_call.args})
        return function_calls

    def execute_functions(self, function_calls: List[dict]) -> List[dict]:
        """
        Execute each requested function with the provided arguments.

        Args:
            function_calls: A list of function call dictionaries.

        Returns:
            A list of results in the form [{function_name: result}, ...].
        """
        results = []
        for fc in function_calls:
            fn_name = list(fc.keys())[0]
            fn_args = fc[fn_name]
            if fn_name in self.functions_map:
                result = self.functions_map[fn_name](**fn_args)
                results.append({fn_name: result})
            else:
                results.append({fn_name: "Error: Function not found."})
        return results

    def send_function_response(self, function_results: List[dict]):
        """
        Send function execution results back to the model as FunctionResponse parts.

        Args:
            function_results: A list of dictionaries with {function_name: result}.

        Returns:
            The model's response object after sending function responses.
        """
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

    def run_query(self, user_input: str) -> str:
        """
        Run a user query against the model. If the model requests function calls, execute them and return the final answer.

        Args:
            user_input: The user's query or prompt.

        Returns:
            A string containing the final textual answer from the model.
        """
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