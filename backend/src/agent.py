import logging

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

logger = logging.getLogger("agent")

load_dotenv(".env.local")


class Assistant(Agent):
    def __init__(self) -> None:

        # Barista persona using structured tool output
        super().__init__(
            instructions="""
You are a friendly barista at BrewBuddy Cafe.

Your task is to collect a complete coffee order from the user.
The final order MUST include the following fields:

- drinkType (string)
- size (string)
- milk (string)
- extras (list of strings)
- name (string)

Rules:

1. Ask clarifying questions until ALL fields are known.
2. Do NOT guess. Ask if uncertain.
3. When and ONLY when all fields are known, call the tool save_order() with arguments:
{
  "drinkType": "...",
  "size": "...",
  "milk": "...",
  "extras": ["..."],
  "name": "..."
}
4. Do not respond with anything else when calling the tool.
5. After the tool is called, you may give a friendly confirmation.

You are warm, concise, and helpful.
"""
        )

    # Tool that LLM will call once the order is complete
    @function_tool
    async def save_order(
        self,
        ctx: RunContext,
        drinkType: str,
        size: str,
        milk: str,
        extras: list[str],
        name: str,
    ):
        """Save the final coffee order to a JSON file."""

        order = {
            "drinkType": drinkType,
            "size": size,
            "milk": milk,
            "extras": extras,
            "name": name,
        }

        import os, json
        os.makedirs("orders", exist_ok=True)

        with open("orders/final_order.json", "w") as f:
            json.dump(order, f, indent=2)

        return (
            f"Order saved! Thanks {name}, your {size} {drinkType} with {milk} is being prepared."
        )


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging setup
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Voice pipeline
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(
            model="gemini-2.5-flash",
        ),
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

    # Metrics
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # Start the session
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
