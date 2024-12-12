# main.py
from gemini_assistant import GeminiAssistant

def add_numbers(a: float, b: float) -> float:
    """Adds two numbers together."""
    print(f"TOOL_USED: Adding {a} and {b}")
    return a + b

def subtract_numbers(a: float, b: float) -> float:
    """Subtracts one number from another."""
    print(f"TOOL_USED: Subtracting {a} and {b}")
    return a - b

def multiply_numbers(a: float, b: float) -> float:
    """Multiplies two numbers."""
    print(f"TOOL_USED: Multiplying {a} and {b}")
    return a * b

def divide_numbers(a: float, b: float) -> float:
    """Divides one number by another. Returns error if division by zero."""
    if b == 0:
        return "Error: Division by zero."
    print(f"TOOL_USED: Dividing {a} by {b}")
    return a / b

if __name__ == "__main__":
    tools = [add_numbers, subtract_numbers, multiply_numbers, divide_numbers]

    # Initialize GeminiAssistant with a list of Python functions.
    # The assistant internally generates an OpenAI-like schema, converts it to Gemini,
    # and prepares the model for parallel function calling.
    assistant = GeminiAssistant(
        functions_list=tools,
        model_name="gemini-2.0-flash-exp",
        generation_config={
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }
    )

    # Run a query that tests the tools with various inputs and prints the final answer.
    query = "Please test the tools with various inputs and summarize the results."
    final_answer = assistant.run_query(query)

    print("\nAssistant Final Answer:")
    print(final_answer)