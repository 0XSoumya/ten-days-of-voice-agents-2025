import json
import logging
import os
import datetime
from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
    function_tool,
)

from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")


# -----------------------
# Load Catalog + Recipes
# -----------------------

def load_catalog():
    with open("shared-data/day7_catalog.json", "r") as f:
        return json.load(f)

def load_recipes():
    with open("shared-data/day7_recipes.json", "r") as f:
        return json.load(f)

CATALOG = load_catalog()
RECIPES = load_recipes()


# ==============================================================
# Helper Functions
# ==============================================================

def find_item(name):
    """Simple keyword-based lookup for catalog items."""
    name = name.lower()
    for item in CATALOG:
        if name in item["name"].lower():
            return item
    return None


# ==============================================================
# Shopping Agent
# ==============================================================

class ShoppingAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="""
You are a friendly grocery & food ordering assistant for InstaCarto — a fictional Instamart-style store.

Your job:
1. Help users order groceries and simple meal ingredients.
2. Add/remove/update items in the cart.
3. Understand requests like:
   - "Add 2 breads"
   - "Remove eggs"
   - "Show my cart"
   - "Ingredients for pasta"
4. NEVER invent items outside the catalog.
5. When the user says "place my order" or "I'm done":
   - Summarize the final cart
   - Save it using the save_order tool
   - Confirm the order has been placed

Be warm, cheerful, helpful — like a real Instamart ordering assistant.
"""
        )

        self.state = {
            "cart": {}  # {item_name: {qty, price}}
        }

    # -----------------------------------------------------------
    # TOOLS
    # -----------------------------------------------------------

    @function_tool
    async def add_item(self, ctx, item_name: str, quantity: int = 1):
        item = find_item(item_name)
        if not item:
            return f"Item '{item_name}' not found in catalog."

        if item["name"] not in self.state["cart"]:
            self.state["cart"][item["name"]] = {
                "quantity": quantity,
                "price": item["price"]
            }
        else:
            self.state["cart"][item["name"]]["quantity"] += quantity

        return f"Added {quantity} × {item['name']} to your cart."

    @function_tool
    async def remove_item(self, ctx, item_name: str):
        if item_name not in self.state["cart"]:
            return f"{item_name} is not in your cart."
        del self.state["cart"][item_name]
        return f"Removed {item_name} from your cart."

    @function_tool
    async def list_cart(self, ctx):
        if not self.state["cart"]:
            return "Your cart is empty."
        items = [f"{v['quantity']} × {k}" for k, v in self.state["cart"].items()]
        return "Your cart contains: " + ", ".join(items)

    @function_tool
    async def add_recipe(self, ctx, recipe_name: str):
        recipe_name = recipe_name.lower()
        if recipe_name not in RECIPES:
            return f"I don't have a recipe for {recipe_name}."

        added_items = []
        for item_name in RECIPES[recipe_name]:
            item = find_item(item_name)
            if item:
                if item["name"] not in self.state["cart"]:
                    self.state["cart"][item["name"]] = {
                        "quantity": 1,
                        "price": item["price"]
                    }
                else:
                    self.state["cart"][item["name"]]["quantity"] += 1
                added_items.append(item["name"])

        return f"For {recipe_name}, I added: " + ", ".join(added_items)

    @function_tool
    async def save_order(self, ctx):
        if not self.state["cart"]:
            return "Cannot place an empty order."

        order = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "items": self.state["cart"],
            "total": sum(
                v["quantity"] * v["price"] for v in self.state["cart"].values()
            )
        }

        os.makedirs("orders", exist_ok=True)
        filename = f"orders/order_{int(datetime.datetime.utcnow().timestamp())}.json"

        with open(filename, "w") as f:
            json.dump(order, f, indent=2)

        return f"Order saved to {filename}"

    # -----------------------------------------------------------
    # MAIN MESSAGE HANDLER
    # -----------------------------------------------------------

    async def on_message(self, ctx, msg):
        text = msg.text.lower()

        # Recipe requests
        for recipe in RECIPES:
            if recipe in text:
                result = await ctx.call_tool(self.add_recipe, recipe_name=recipe)
                await ctx.send_message(result)
                return

        # Add item
        if "add" in text or "get me" in text:
            words = text.split()
            quantity = 1

            # Try to extract quantity
            for w in words:
                if w.isdigit():
                    quantity = int(w)

            # Try to find matching catalog item
            for item in CATALOG:
                if item["name"].lower() in text:
                    result = await ctx.call_tool(
                        self.add_item, item_name=item["name"], quantity=quantity
                    )
                    await ctx.send_message(result)
                    return

        # Remove item
        if "remove" in text or "delete" in text:
            for item in list(self.state["cart"].keys()):
                if item.lower() in text:
                    result = await ctx.call_tool(self.remove_item, item_name=item)
                    await ctx.send_message(result)
                    return

        # Show cart
        if "cart" in text or "what's in my cart" in text:
            result = await ctx.call_tool(self.list_cart)
            await ctx.send_message(result)
            return

        # Place order
        if any(p in text for p in ["place order", "that's all", "checkout", "i'm done"]):
            summary = await ctx.call_tool(self.list_cart)
            await ctx.send_message("Sure! Here's your final cart:")
            await ctx.send_message(summary)

            saved = await ctx.call_tool(self.save_order)
            await ctx.send_message(saved)
            await ctx.send_message("Your order has been placed 🎉")
            await ctx.session.close()
            return

        # Default
        await ctx.send_message(
            "How can I help? You can ask me to add items, remove items, show your cart, or get ingredients for a recipe."
        )


# ==============================================================
# LiveKit Setup
# ==============================================================

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-matthew",
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    usage = metrics.UsageCollector()
    @session.on("metrics_collected")
    def _m(ev: MetricsCollectedEvent):
        usage.collect(ev.metrics)

    await session.start(
        agent=ShoppingAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
