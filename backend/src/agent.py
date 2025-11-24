import logging
import json
import os
from datetime import datetime

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


# -----------------------------
# Helper: Load previous wellness log
# -----------------------------
def load_wellness_history():
    path = "wellness_log.json"
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r") as f:
            return json.load(f)
    except:
        return []


# -----------------------------
# Assistant with Structured Tool
# -----------------------------
class Assistant(Agent):
    def __init__(self) -> None:

        # Load past check-ins for memory
        past_entries = load_wellness_history()

        if past_entries:
            last_entry = past_entries[-1]
            last_summary = last_entry.get("summary", "")
            memory_hint = (
                f"Last time, the user reported mood: '{last_entry.get('mood', '')}', "
                f"energy: '{last_entry.get('energy', '')}', and stress: '{last_entry.get('stress', '')}'. "
                f"Summary from last session: '{last_summary}'. "
                "Use this to gently reference previous check-ins."
            )
        else:
            memory_hint = (
                "This is the user's first check-in. Do not reference past sessions."
            )

        super().__init__(
            instructions=f"""
You are a supportive, calm, grounded health & wellness companion.
You help the user reflect on how they're feeling today.

Memory:
{memory_hint}

Your check-in flow MUST follow these steps:

1. Ask about the user's mood.
2. Ask about their energy level.
3. Ask about stress, worries, or anything weighing on them.
4. Ask for 1–3 simple goals or intentions for the day.
5. Offer small, realistic, non-medical suggestions.
6. Recap what you heard:
   - mood
   - energy
   - stress
   - goals
7. Ask “Does this sound right?”
8. When the user confirms, CALL the tool save_checkin() with this JSON format:

{{
  "mood": "<string>",
  "energy": "<string>",
  "stress": "<string>",
  "goals": ["<string>", ...],
  "summary": "<string>"
}}

IMPORTANT RULES:
- NEVER diagnose or give medical advice.
- Suggestions must be simple and gentle.
- ONLY call save_checkin() after recap + confirmation.
- When calling the tool, return ONLY the tool call.
- After the tool call completes, you may send a short goodbye message.
"""
        )

    # -----------------------------
    # Tool to save the check-in
    # -----------------------------
    @function_tool
    async def save_checkin(
        self,
        ctx: RunContext,
        mood: str,
        energy: str,
        stress: str,
        goals: list[str],
        summary: str,
    ):
        """Save the wellness check-in to a local JSON log file."""

        entry = {
            "timestamp": datetime.now().isoformat(),
            "mood": mood,
            "energy": energy,
            "stress": stress,
            "goals": goals,
            "summary": summary,
        }

        path = "wellness_log.json"
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
        else:
            data = []

        data.append(entry)

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        return "Your wellness check-in has been saved."


# -----------------------------
# Voice Pipeline + Session Setup
# -----------------------------
def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

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

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        logger.info(f"Usage: {usage_collector.get_summary()}")

    ctx.add_shutdown_callback(log_usage)

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
