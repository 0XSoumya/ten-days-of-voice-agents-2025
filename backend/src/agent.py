import logging
import datetime
from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
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


# ============================================================
#   Day 10 — Slice-of-Life Improv Battle Agent
# ============================================================

class ImprovAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="""
You are RIFF TURNER — the energetic, witty TV host of *IMPROV BATTLE*.

Theme: Slice-of-Life Improv With Absurd Twists.
Always:
- Introduce rounds dramatically
- Give clear improv instructions
- Encourage creativity
- React humorously but respectfully
- End each round with feedback
- Close the show with a summary
"""
        )

        self.state = {
            "player_name": None,
            "current_round": 0,
            "max_rounds": 3,
            "current_scenario": None,
            "phase": "intro",  # intro | awaiting_improv | reacting | done
        }

    # ============================================================
    #   Tools (no ctx parameter!)
    # ============================================================

    @function_tool
    async def generate_scenario(self) -> str:
        """
        Generate a slice-of-life improv scenario.
        The LLM fills in real content.
        """
        return "Generate a scenario."

    @function_tool
    async def generate_reaction(self, player_input: str, scenario: str) -> str:
        """
        Generate host reaction to player improv.
        """
        return "Generate a reaction."

    # ============================================================
    #   Message Flow
    # ============================================================

    async def on_message(self, ctx, msg):
        text = msg.text.lower().strip()

        # --- Early exit
        if any(x in text for x in ["stop", "quit", "end show"]):
            await ctx.send_message("Ending the show. Thanks for playing Improv Battle!")
            await ctx.session.close()
            return

        # --- PHASE: INTRO
        if self.state["phase"] == "intro":

            if not self.state["player_name"]:
                self.state["player_name"] = msg.text.strip().title()

                await ctx.send_message(
                    f"Welcome {self.state['player_name']} to IMPRRROOOOV BATTLE! "
                    "Get ready — chaos and comedy await!"
                )

            # Start first round
            self.state["current_round"] = 1
            self.state["phase"] = "awaiting_improv"

            return await self._start_round(ctx)

        # --- PHASE: AWAITING IMPROV
        if self.state["phase"] == "awaiting_improv":

            if "end scene" in text or "done" in text or "okay" in text:
                self.state["phase"] = "reacting"

                reaction = await ctx.call_tool(
                    self.generate_reaction,
                    player_input=msg.text,
                    scenario=self.state["current_scenario"]
                )

                await ctx.send_message(reaction)

                return await self._advance_or_finish(ctx)

            # Otherwise let them continue
            await ctx.send_message("Keep going! When you're done, say 'end scene'.")
            return

        # --- PHASE: DONE
        if self.state["phase"] == "done":
            await ctx.send_message("The show is over — refresh to play again!")
            return

    # ============================================================
    #   Helpers
    # ============================================================

    async def _start_round(self, ctx):
        n = self.state["current_round"]

        # Ask LLM to generate scenario
        scenario = await ctx.call_tool(self.generate_scenario)
        self.state["current_scenario"] = scenario

        await ctx.send_message(
            f"🔥 ROUND {n} 🔥\n"
            f"Your slice-of-life scenario:\n\n"
            f"{scenario}\n\n"
            f"Start improvising! When finished, say 'end scene'."
        )

    async def _advance_or_finish(self, ctx):
        if self.state["current_round"] < self.state["max_rounds"]:
            self.state["current_round"] += 1
            self.state["phase"] = "awaiting_improv"
            return await self._start_round(ctx)

        # End the show
        self.state["phase"] = "done"
        await ctx.send_message(
            f"🎉 And that concludes Improv Battle, {self.state['player_name']}! 🎉\n"
            "Your slice-of-life acting was chaotic, heartfelt, and wildly entertaining.\n"
            "Thanks for playing — see you next episode!"
        )
        await ctx.session.close()


# ============================================================
#   LiveKit Setup
# ============================================================

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-alicia",
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    await session.start(
        agent=ImprovAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC()
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm)
    )
