import os
import asyncio
from dotenv import load_dotenv
from google.genai import types
from google.adk.agents.llm_agent import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters

load_dotenv()

# Ensure TARGET_FOLDER_PATH is an absolute path for the MCP server.
def setup_target_directory():
    """Setup and validate the target directory."""
    target_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_folder")
    try:
        os.makedirs(target_path, exist_ok=True)
        # Test if directory is writable
        test_file = os.path.join(target_path, '.write_test')
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        return target_path
    except (OSError, IOError) as e:
        print(f"Error setting up target directory: {e}")
        print("Please ensure you have write permissions for the directory.")
        raise

TARGET_FOLDER_PATH = setup_target_directory()

# --- Step 1: Agent Definition ---
async def get_agent_async():
  """Creates an ADK Agent equipped with tools from the MCP Server."""
  toolset = MCPToolset(
      # Use StdioServerParameters for local process communication
      connection_params=StdioServerParameters(
          command='npx', # Command to run the server
          args=["-y",    # Arguments for the command
                "@modelcontextprotocol/server-filesystem",
                os.path.abspath(TARGET_FOLDER_PATH)],
      ),
      # tool_filter=['read_file', 'list_directory'] # Optional: filter specific tools
      # For remote servers, you would use SseServerParams instead:
      # connection_params=SseServerParams(url="http://remote-server:port/path", headers={...})
  )

  # Use in an agent
  root_agent = LlmAgent(
      model='gemini-2.0-flash', # Adjust model name if needed based on availability
      name='mcp_filesystem_agent',
      instruction='Help the user manage their files. You can list files, read files, etc. Dont ask the user for name of directory you are working with, you already know it, just do the tasks user is asking for.',
      tools=[toolset], # Provide the MCP tools to the ADK agent
  )
  return root_agent, toolset

# --- Step 2: Main Execution Logic ---
async def interactive_agent_session():
    """Runs an interactive session with the agent, accepting queries from terminal."""
    session_service = InMemorySessionService()
    session = await session_service.create_session(
        state={}, app_name='mcp_filesystem_app', user_id='user_fs'
    )

    # Initialize agent and runner once
    root_agent, toolset = await get_agent_async()
    runner = Runner(
        app_name='mcp_filesystem_app',
        agent=root_agent,
        session_service=session_service,
    )

    print("\nInteractive File System Agent (type 'exit' to quit)")
    print("=" * 50)

    try:
        while True:
            # Get user input
            query = input("\nEnter your query: ").strip()
            
            if query.lower() == 'exit':
                break

            print(f"\nProcessing: '{query}'")
            content = types.Content(role='user', parts=[types.Part(text=query)])

            # Process the query
            async for event in runner.run_async(
                session_id=session.id,
                user_id=session.user_id,
                new_message=content
            ):
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, 'text') and part.text:
                            print(f"Agent: {part.text}")

    finally:
        print("\nClosing MCP server connection...")
        await toolset.close()
        print("Session ended.")

if __name__ == '__main__':
    try:
        asyncio.run(interactive_agent_session())
    except KeyboardInterrupt:
        print("\nSession terminated by user.")
    except Exception as e:
        print(f"An error occurred: {e}")