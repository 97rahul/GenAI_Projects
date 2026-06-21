import random
from fastmcp import FastMCP


mcp = FastMCP("Demo Server")

@mcp.tool()
def random_number():
    """Returns a random number between 1 and 100."""
    return random.randint(1, 100)

@mcp.tool()
def add_numbers(a: int, b: int):
    """Adds two numbers and returns the result."""
    return a + b

@mcp.tool()
def mul_numbers(a: int, b: int):
    """Multiplies two numbers and returns the result."""
    return a * b

@mcp.tool()
def div_numbers(a: int, b: int):
    """Divides two numbers and returns the result."""
    return a / b

if __name__ == "__main__":
    mcp.run()