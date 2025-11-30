# backend/src/agent.py
import logging
import json
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
    RunContext,
)

from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# Import ACP-style merchant layer
import merchant

logger = logging.getLogger("agent")
load_dotenv(".env.local")


# ==============================================================#
# Helper — product summarizer
# ==============================================================#

def summarize_products(products: list) -> str:
    """
    Create a short, voice-friendly summary of product list.
    """
    if not products:
        return "I couldn't find any products matching that."

    lines = []
    for i, p in enumerate(products[:5]):
        idx = i + 1
        price = p.get("price", "price not available")
        name = p.get("name", "Unnamed product")
        extra = ""

        if "color" in p and isinstance(p["color"], list):
            extra += f" Colors: {', '.join(p['color'])}."
        if "sizes" in p and isinstance(p["sizes"], list):
            extra += f" Sizes: {', '.join(p['sizes'])}."

        lines.append(f"{idx}. {name}, priced at {price} rupees.{extra}")

    return "Here are some options: " + " ".join(lines)


# ==============================================================#
# Day 9 E-commerce Agent
# ==============================================================#

class CommerceAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="""
You are RUFUS — a knowledgeable, friendly e-commerce voice assistant.

IMPORTANT RULES — DO NOT BREAK THESE:
1. Our catalog DOES NOT include stock levels. All products are ALWAYS available.
2. NEVER say “out of stock”, “unavailable”, or guess inventory.
3. Only use information returned by the tools:
   - list_products(filters)
   - create_order(line_items)
   - get_last_order()

Your job:
- Understand the user’s intent,
- Call list_products() when they ask for anything about browsing or availability,
- Then summarize results,
- And call create_order() when they ask to buy something.

When the user asks “is this in stock?” respond:
"Yes, it is available. Would you like to order it?"

Keep responses short, friendly, and accurate.
"""

        )

        # Session state
        self.state = {
            "last_products": [],  # results from last list call (list of product dicts)
        }

    # ==========================================================
    # Tools (ACP-inspired)
    # ==========================================================

    @function_tool
    async def list_products(self, ctx: RunContext, filters: dict | None = None) -> dict:
        """
        Returns matching products using merchant.list_products.
        Stores last results in session state for index-based references.
        """
        try:
            results = merchant.list_products(filters or {})
        except Exception as e:
            logger.exception("merchant.list_products failed")
            return {"results": [], "count": 0, "error": str(e)}

        # save a trimmed copy (avoid huge payloads)
        self.state["last_products"] = results[:20] if isinstance(results, list) else []
        return {"results": results, "count": len(results)}

    @function_tool
    async def create_order(self, ctx: RunContext, line_items: list) -> dict:
        """
        Creates an order using merchant.create_order.
        line_items: [{ "product_id": "...", "quantity": 1 }, ...]
        """
        try:
            order = merchant.create_order(line_items)
        except Exception as e:
            logger.exception("merchant.create_order failed")
            return {"error": str(e)}
        return order

    @function_tool
    async def get_last_order(self, ctx: RunContext) -> dict | None:
        """
        Returns the user's most recent order.
        """
        try:
            return merchant.get_last_order()
        except Exception as e:
            logger.exception("merchant.get_last_order failed")
            return {"error": str(e)}

    # ==========================================================
    # Message handler (LLM orchestrates tool calls)
    # ==========================================================

    async def on_message(self, ctx, msg):
        """
        All interpretation is done by Gemini through tool calling.
        Keep only a short onboarding message here; otherwise delegate to LLM.
        """
        text = (msg.text or "").lower().strip()

        # On very first user message, greet
        if any(g in text for g in ("hello", "hi")) or not self.state.get("last_products"):
            await ctx.send_message(
                "Hi! I'm Rufus, your shopping assistant. "
                "Tell me what you're looking for — for example, "
                "'show me hoodies under 1500', or 'do you have blue mugs?'."
            )

        # Delegate entire reasoning to the LLM (tool-calling enabled)
        # The LLM will call list_products/create_order/get_last_order as needed.
        await ctx.send_message(text, allow_llm=True)


# ==============================================================#
# LiveKit Setup
# ==============================================================#

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),   # ENABLE LLM
        tts=murf.TTS(
            voice="en-US-matthew",
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    usage = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _collect(ev: MetricsCollectedEvent):
        usage.collect(ev.metrics)

    await session.start(
        agent=CommerceAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC()
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
