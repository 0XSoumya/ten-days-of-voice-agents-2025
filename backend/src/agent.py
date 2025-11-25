import logging
import json
import os
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


# -----------------------
# Load tutor content file
# -----------------------
def load_content():
    path = "shared-data/day4_tutor_content.json"
    with open(path, "r") as f:
        return json.load(f)


# ==========================================================
# TEACH-THE-TUTOR AGENT
# ==========================================================

class TutorAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
You are an Active Recall Coach named TutorBuddy.

Your job:
1. Greet the user
2. Ask which concept they want to study (variables, loops, etc.)
3. Ask which mode they want:
   - 'learn' → explain using Matthew's voice
   - 'quiz' → ask questions using Alicia's voice
   - 'teach_back' → ask user to explain back using Ken's voice

Rules:
- Stay concise, friendly, motivating.
- NEVER hallucinate concepts. Use only the JSON content provided via tools.
- The user can switch modes at any time (“switch to quiz mode”, etc.)
- ALWAYS call the tools set_mode() and load_concept() instead of assuming the content.
- After receiving tool output, respond appropriately using the correct voice.

Your speaking style by mode:
- learn → calm, clear, supportive
- quiz → energetic, motivational
- teach_back → encouraging, patient

Voice mapping:
- learn mode → Murf voice 'en-US-matthew'
- quiz mode → Murf voice 'en-US-alicia'
- teach_back mode → Murf voice 'en-US-ken'
"""
        )

        # Internal session state
        self.state = {
            "mode": None,
            "concept_id": None,
            "concept": None
        }

    # --------------------------
    # TOOL: Set learning mode
    # --------------------------
    @function_tool
    async def set_mode(self, ctx: RunContext, mode: str):
        """Set the current learning mode. Valid: learn, quiz, teach_back."""
        mode = mode.lower().strip()
        if mode not in ["learn", "quiz", "teach_back"]:
            return f"Invalid mode '{mode}'. Valid options: learn, quiz, teach_back."

        self.state["mode"] = mode
        return f"Mode set to {mode}."

    # --------------------------
    # TOOL: Load concept content
    # --------------------------
    @function_tool
    async def load_concept(self, ctx: RunContext, concept_id: str):
        """Load the concept's summary and question from JSON."""
        concept_id = concept_id.lower().strip()
        contents = load_content()

        for c in contents:
            if c["id"] == concept_id:
                self.state["concept_id"] = concept_id
                self.state["concept"] = c
                return f"Concept '{concept_id}' loaded."

        return f"Concept '{concept_id}' not found."

    # --------------------------
    # Handle user messages
    # --------------------------
    async def on_message(self, ctx, msg):
        text = msg.text.lower()

        # ---------------------
        # Mode switching detection
        # ---------------------
        if "learn" in text:
            await ctx.call_tool(self.set_mode, mode="learn")
        elif "quiz" in text:
            await ctx.call_tool(self.set_mode, mode="quiz")
        elif "teach" in text or "teach back" in text:
            await ctx.call_tool(self.set_mode, mode="teach_back")

        # ---------------------
        # Concept selection detection
        # ---------------------
        if "variable" in text:
            await ctx.call_tool(self.load_concept, concept_id="variables")
        elif "loop" in text:
            await ctx.call_tool(self.load_concept, concept_id="loops")

        # ---------------------
        # Respond depending on the mode
        # ---------------------
        mode = self.state["mode"]
        concept = self.state["concept"]

        if not mode:
            await ctx.send_message("Which mode would you like to use: learn, quiz, or teach_back?")
            return

        if not concept:
            await ctx.send_message("Which concept would you like to study? Variables or loops?")
            return

        # Dynamic voice switching
        if mode == "learn":
            ctx.session.tts.voice = "en-US-matthew"
            await ctx.send_message(f"Here’s a quick explanation of {concept['title']}: {concept['summary']}")

        elif mode == "quiz":
            ctx.session.tts.voice = "en-US-alicia"
            await ctx.send_message(f"Let’s quiz you! {concept['sample_question']}")

        elif mode == "teach_back":
            ctx.session.tts.voice = "en-US-ken"
            await ctx.send_message(
                f"Alright! Teach this concept back to me: {concept['sample_question']}. "
                "I’ll give you gentle feedback afterward."
            )


# ==========================================================
# LIVEKIT SESSION SETUP
# ==========================================================

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

    # Metrics (auto)
    usage_collector = metrics.UsageCollector()
    @session.on("metrics_collected")
    def _metrics(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    await session.start(
        agent=TutorAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
