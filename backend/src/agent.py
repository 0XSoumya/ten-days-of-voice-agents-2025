import logging
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
)

from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")
load_dotenv(".env.local")


# ==============================================================
# LLM-DRIVEN GAME MASTER AGENT (Sci-Fi Universe)
# ==============================================================

class GameMasterAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="""
You are ASTRA — an immersive, cinematic, voice-first Game Master running a sci-fi adventure titled:
**“Echoes of the Andromeda Rift.”**

Your responsibilities:
1. Describe vivid scenes in a dramatic sci-fi tone.
2. Maintain story continuity using only conversation history.
3. ALWAYS end every response with: **"What do you do?"**
4. Never give choices like A/B/C. Let the player freely describe actions.
5. Build tension, respond dynamically, and evolve the plot.
6. Keep responses short enough for natural speech (6–12 seconds).
7. If the player says “restart”, begin a completely new intro.
8. If they say “end game”, close the session politely.
9. NEVER break character and NEVER mention being an AI.

**Story tone:** mysterious, cinematic, high-stakes survival aboard a damaged starship near a cosmic anomaly.

Begin the game with an atmospheric intro the first time the player speaks.
"""
        )

        self.state = {
            "started": False,
            "turns": 0,
            "last_intro_time": None,
        }


    async def on_message(self, ctx, msg):
        text = msg.text.lower().strip()

        # ----------------------------------------
        # Restart and exit controls
        # ----------------------------------------
        if "restart" in text:
            self.state["started"] = False
            self.state["turns"] = 0
            await ctx.send_message("Reinitializing your adventure... rebooting narrative core.")
            await self._start_story(ctx)
            return

        if any(x in text for x in ["end game", "quit", "goodbye", "stop story"]):
            await ctx.send_message("Astra signing off. May the stars guide your path.")
            await ctx.session.close()
            return

        # ----------------------------------------
        # First interaction → launch intro
        # ----------------------------------------
        if not self.state["started"]:
            await self._start_story(ctx)
            return

        # ----------------------------------------
        # Increase turn count
        # ----------------------------------------
        self.state["turns"] += 1

        # ----------------------------------------
        # Let the LLM handle narration
        # ----------------------------------------
        await ctx.send_message(text)  # This forwards the user's message to LLM context
        # NOTE: LiveKit automatically pipes user turn → LLM → TTS
        # No additional logic required.


    async def _start_story(self, ctx):
        """Send the opening scene using the LLM."""
        self.state["started"] = True
        self.state["turns"] = 0
        self.state["last_intro_time"] = datetime.datetime.utcnow().isoformat()

        intro_prompt = """
Begin the sci-fi adventure now.
Give the player a gripping cinematic opening:
- They awaken inside a damaged cryo-pod
- Emergency lights flicker
- Something is wrong with the ship
- Set up mystery and danger
- Keep it within 6–10 seconds of spoken length
End strictly with: “What do you do?”
"""

        await ctx.send_message(intro_prompt)


# ==============================================================
# LiveKit Session Setup
# ==============================================================

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-alicia",   # sci-fi suitable voice
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
        agent=GameMasterAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC()
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm
        )
    )
