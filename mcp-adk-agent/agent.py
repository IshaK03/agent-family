# ./adk_agent_samples/mcp_agent/agent.py
import os # Required for path operations
from google.adk.agents import LlmAgent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters

# # It's good practice to define paths dynamically if possible,
# # or ensure the user understands the need for an ABSOLUTE path.
# # For this example, we'll construct a path relative to this file,
# # assuming '/path/to/your/folder' is in the same directory as agent.py.
# # REPLACE THIS with an actual absolute path if needed for your setup.
TARGET_FOLDER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_folder")
# # Ensure TARGET_FOLDER_PATH is an absolute path for the MCP server.
# # If you created ./adk_agent_samples/mcp_agent/your_folder,

# # Create a test folder in the same directory as the script
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# TARGET_FOLDER_PATH = os.path.join(BASE_DIR, "test_folder")

# # Ensure the target directory exists
# os.makedirs(TARGET_FOLDER_PATH, exist_ok=True)

root_agent = LlmAgent(
    model='gemini-2.0-flash',
    name='filesystem_assistant_agent',
    instruction='Help the user manage their files. You can list files, read files, etc. Dont ask the user for name of directory you are working with, you already know it, just do the tasks user is asking for.',
    tools=[
        MCPToolset(
            connection_params=StdioServerParameters(
                command='npx',
                args=[
                    "-y",  # Argument for npx to auto-confirm install
                    "@modelcontextprotocol/server-filesystem",
                    # IMPORTANT: This MUST be an ABSOLUTE path to a folder the
                    # npx process can access.
                    # Replace with a valid absolute path on your system.
                    # For example: "/Users/youruser/accessible_mcp_files"
                    # or use a dynamically constructed absolute path:
                    os.path.abspath(TARGET_FOLDER_PATH),
                ],
            ),
            # Optional: Filter which tools from the MCP server are exposed
            # tool_filter=['list_directory', 'read_file']
        )
    ],
)

'''
Run using: "adk run mcp-adk-agent" from the agent-family directory
'''