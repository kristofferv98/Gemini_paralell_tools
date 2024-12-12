# tool_generators.py
from tool_converters.schema_converter import generate_all_schemas


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
    """Divides one number by another."""
    if b == 0:
        return "Error: Division by zero."
    print(f"TOOL_USED: Dividing {a} by {b}")
    return a / b

tools = [add_numbers, subtract_numbers, multiply_numbers, divide_numbers]

# Generate all schemas
all_schemas = generate_all_schemas(tools)

# Example: get the Anthropic schema
anthropic_schema = all_schemas["anthropic"]
print("Anthropic Schema:")
print(anthropic_schema)

# Example: get the Gemini schema
gemini_schema = all_schemas["gemini"]
print("Gemini Schema:")
print(gemini_schema)

# Example: get the OpenAI schema
openai_schema = all_schemas["openai"]
print("OpenAI Schema:")
print(openai_schema)
