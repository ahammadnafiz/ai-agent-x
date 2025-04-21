import os
import asyncio
import argparse
import sys
from langchain_google_genai import ChatGoogleGenerativeAI
from browser_use import Agent, Browser, BrowserContextConfig, BrowserConfig
from browser_use.browser.browser import BrowserContext
from pydantic import SecretStr
from dotenv import load_dotenv

async def setup_browser(headless: bool = False):
    """Initialize and configure the browser"""
    browser = Browser(
        config=BrowserConfig(
            headless=headless,
        ),
    )
    context_config = BrowserContextConfig(
        wait_for_network_idle_page_load_time=5.0,
        highlight_elements=True,
        save_recording_path="./recordings",
    )
    return browser, BrowserContext(browser=browser, config=context_config)

async def agent_loop(llm, browser_context, query, initial_url=None):
    """Run agent loop with optional initial URL"""
    # Set up initial actions if URL is provided
    initial_actions = None
    if initial_url:
        initial_actions = [
            {"open_tab": {"url": initial_url}},
        ]

    agent = Agent(
        task=query,
        llm=llm,
        browser_context=browser_context,
        use_vision=True,
        generate_gif=True,
        initial_actions=initial_actions,
    )

    # Start Agent and browser
    result = await agent.run()

    return result.final_result() if result else None

async def main():
    try:
        # Load environment variables
        load_dotenv()

        # Disable telemetry - set this before importing browser_use
        os.environ["ANONYMIZED_TELEMETRY"] = "false"
        os.environ["BROWSER_USE_TELEMETRY_OPT_OUT"] = "1"  # Additional telemetry opt-out

        # --- Argument Parsing ---
        parser = argparse.ArgumentParser(
            description="Run Gemini agent with browser interaction."
        )
        parser.add_argument(
            "--model",
            type=str,
            default="gemini-2.5-flash-preview-04-17",
            help="The Gemini model to use.",
        )
        parser.add_argument(
            "--headless",
            action="store_true",
            help="Run the browser in headless mode.",
        )
        parser.add_argument(
            "--url",
            type=str,
            help="Starting URL for the browser to navigate to before user query.",
        )
        parser.add_argument(
            "--query",
            type=str,
            help="The query to process.",
        )
        args = parser.parse_args()
        # --- End Argument Parsing ---

        # Check for API key in environment variable
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("ERROR: GEMINI_API_KEY environment variable not set.")
            print("Please set your API key using: export GEMINI_API_KEY=your_api_key")
            print("Or add it to your .env file as GEMINI_API_KEY=your_api_key")
            sys.exit(1)

        # Initialize the Gemini model
        llm = ChatGoogleGenerativeAI(
            model=args.model,
            api_key=api_key,
        )

        # Setup browser
        browser, context = await setup_browser(headless=args.headless)

        if args.query:
            result = await agent_loop(llm, context, args.query, initial_url=args.url)
            print(result)
        else:
            # Get search queries from user
            while True:
                try:
                    # Get user input and check for exit commands
                    user_input = input("\nEnter your prompt (or 'quit' to exit): ")
                    if user_input.lower() in ["quit", "exit", "q"]:
                        break

                    # Process the prompt and run agent loop with initial URL if provided
                    result = await agent_loop(
                        llm, context, user_input, initial_url=args.url
                    )

                    # Clear URL after first use to avoid reopening same URL in subsequent queries
                    args.url = None

                    # Display the final result with clear formatting
                    print("\n📊 Search Results:")
                    print("=" * 50)
                    print(result if result else "No results found")
                    print("=" * 50)

                except KeyboardInterrupt:
                    print("\nExiting...")
                    break
                except Exception as e:
                    print(f"\nError occurred: {e}")

        print("Closing browser")
        # Ensure browser is closed properly
        await context.close()  # Close context first
        await browser.close()  # Then close browser

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Make sure everything is cleaned up
        # This final section helps prevent "coroutine was never awaited" warnings
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        for task in tasks:
            task.cancel()


if __name__ == "__main__":
    # Use run_until_complete instead of run for better cleanup
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main())
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
