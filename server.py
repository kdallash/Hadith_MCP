from mcp.server.fastmcp import FastMCP

mcp = FastMCP("test-server")

@mcp.tool()
def shout(text: str) -> str:
    """Returns the input text in uppercase."""
    return text.upper()

if __name__ == "__main__":
    mcp.run()