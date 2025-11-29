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

def find_item(name: str):
    """Simple keyword lookup inside catalog."""
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
            instructions=f"""
You are a friendly grocery & food ordering assistant for InstaCarto — a fictional Instamart-style store.

Your job:
1. Help users order items from the catalog.
2. Use only the provided tools for modifying the cart.
3. Understand user requests such as:
   - "Add 2 breads"
   - "Remove eggs"
   - "Show my cart"
   - "Ingredients for pasta"
4. Use recipe → add_recipe tool for recipe-based requests.
5. When the user says: "place order", "checkout", "I'm done":
   - Call list_cart
   - Then call save_order
   - Confirm order completion

NEVER invent items outside this catalog:
{", ".join(item['name'] for item in CATALOG)}

Available recipes:
{", ".join(RECIPES.keys())}

Stay cheerful, warm and helpful — like a real Instamart assistant.
"""
        )

        self.state = {
            "cart": {}  # {item_name: {quantity, price}}
        }

    # -----------------------------------------------------------
    # TOOLS (NO ctx param — Gemini SAFE)
    # -----------------------------------------------------------

    @function_tool
    async def add_item(self, item_name: str, quantity: int = 1) -> str:
        """Add an item to the cart."""

        item = find_item(item_name)
        if not item:
            return f"Item '{item_name}' not found in catalog."

        name = item["name"]

        if name not in self.state["cart"]:
            self.state["cart"][name] = {
                "quantity": quantity,
                "price": item["price"]
            }
        else:
            self.state["cart"][name]["quantity"] += quantity

        return f"Added {quantity} × {name} to your cart."

    @function_tool
    async def remove_item(self, item_name: str) -> str:
        """Remove item from cart."""

        key = next((k for k in self.state["cart"] if k.lower() == item_name.lower()), None)
        if not key:
            return f"{item_name} is not in your cart."

        del self.state["cart"][key]
        return f"Removed {key} from your cart."

    @function_tool
    async def update_quantity(self, item_name: str, new_quantity: int) -> str:
        """Update quantity of an existing item."""

        key = next((k for k in self.state["cart"] if k.lower() == item_name.lower()), None)
        if not key:
            return f"{item_name} is not in your cart."

        if new_quantity <= 0:
            del self.state["cart"][key]
            return f"Removed {key} from your cart."

        self.state["cart"][key]["quantity"] = new_quantity
        return f"Updated {key} quantity to {new_quantity}."

    @function_tool
    async def list_cart(self) -> str:
        """List items in the cart with total."""

        if not self.state["cart"]:
            return "Your cart is empty."

        lines = []
        total = 0

        for name, info in self.state["cart"].items():
            qty = info["quantity"]
            price = info["price"]
            subtotal = qty * price
            total += subtotal
            lines.append(f"{qty} × {name} — ₹{subtotal}")

        lines.append(f"\nTotal: ₹{total}")
        return "\n".join(lines)

    @function_tool
    async def add_recipe(self, recipe_name: str) -> str:
        """Add recipe-based ingredients."""

        key = next((k for k in RECIPES if k.lower() == recipe_name.lower()), None)
        if not key:
            return f"I don't have a recipe named '{recipe_name}'."

        added = []
        for ingredient in RECIPES[key]:
            item = find_item(ingredient)
            if item:
                name = item["name"]
                if name not in self.state["cart"]:
                    self.state["cart"][name] = {"quantity": 1, "price": item["price"]}
                else:
                    self.state["cart"][name]["quantity"] += 1
                added.append(name)

        return f"For {key}, I added: " + ", ".join(added)

    @function_tool
    async def save_order(self) -> str:
        """Save final order to a JSON file."""

        if not self.state["cart"]:
            return "Your cart is empty — cannot place order."

        order = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "items": self.state["cart"],
            "total": sum(v["quantity"] * v["price"] for v in self.state["cart"].values())
        }

        os.makedirs("orders", exist_ok=True)
        filename = f"orders/order_{int(datetime.datetime.utcnow().timestamp())}.json"

        with open(filename, "w") as f:
            json.dump(order, f, indent=2)

        # Clear cart after placing order
        self.state["cart"] = {}

        return f"Your order has been placed! Saved to {filename}"

    # -----------------------------------------------------------
    # MESSAGE HANDLER (LLM routing logic)
    # -----------------------------------------------------------

    async def on_message(self, ctx, msg):
        text = msg.text.lower()

        # Recipe detection
        for recipe in RECIPES:
            if recipe.lower() in text:
                result = await ctx.call_tool(self.add_recipe, recipe_name=recipe)
                await ctx.send_message(result)
                return

        # Add items
        if any(x in text for x in ["add", "get me", "i want"]):
            quantity = 1
            words = text.split()

            for w in words:
                if w.isdigit():
                    quantity = int(w)

            for item in CATALOG:
                if item["name"].lower() in text:
                    result = await ctx.call_tool(
                        self.add_item,
                        item_name=item["name"],
                        quantity=quantity
                    )
                    await ctx.send_message(result)
                    return

        # Remove items
        if "remove" in text or "delete" in text:
            for name in list(self.state["cart"].keys()):
                if name.lower() in text:
                    result = await ctx.call_tool(self.remove_item, item_name=name)
                    await ctx.send_message(result)
                    return

        # Show cart
        if "cart" in text or "what's in my cart" in text:
            result = await ctx.call_tool(self.list_cart)
            await ctx.send_message(result)
            return

        # Place order
        if any(x in text for x in ["place order", "checkout", "that's all", "i'm done"]):
            summary = await ctx.call_tool(self.list_cart)
            await ctx.send_message("Here is your final cart:\n" + summary)

            saved = await ctx.call_tool(self.save_order)
            await ctx.send_message(saved)

            await ctx.session.close()
            return

        # Default fallback
        await ctx.send_message(
            "I can help you order groceries! Try saying things like 'add 2 breads', "
            "'ingredients for pasta', or 'show my cart'."
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
