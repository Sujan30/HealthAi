from langchain_openai import ChatOpenAI
from browser_use import Agent
import asyncio
from dotenv import load_dotenv
from main import get_products, prompt

load_dotenv()

products = []

def initializeProducts():
    global products
    products = get_products(prompt)
    return products


async def productsSearch():
    global products
    links = []

    for product in products:
        try:
            # Instantiate the agent within the loop to create a new browser context
            agent = Agent(
                task=f"""
                Go to google and search for '{product.lstrip('-').strip()}'.  
                Once the search results page loads, click on the "Shopping" tab or link.
                On the Google Shopping results page, find the first product listing.
                Click on the first product listing to open its details page.
                Extract the URL of the product details page.  Do not close the browser or the page till this is done.
                """,
                llm=ChatOpenAI(model="gpt-4o")
            )
            result = await agent.run()
            links.append(result)
            print(f"Link for {product}: {result}")

            await asyncio.sleep(5)  # Wait for 5 seconds - adjust as needed

        except Exception as e:
            print(f"Error processing {product}: {e}")
            # Optionally, re-raise the exception for debugging purposes
            # raise

    return links

if __name__ == "__main__":
    try:
        initializeProducts()
        asyncio.run(productsSearch())
    except Exception as e:
        print(f"An error occurred: {e}")
        