import logging
import sqlite3
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


# -------------------------
# Database Helper
# -------------------------
def load_case_by_name(name: str):
    conn = sqlite3.connect("fraud_cases.db")
    cursor = conn.cursor()
    row = cursor.execute(
        "SELECT * FROM fraud_cases WHERE userName LIKE ?", (name,)
    ).fetchone()
    conn.close()
    return row


def update_case_status(case_id: int, status: str, note: str):
    conn = sqlite3.connect("fraud_cases.db")
    cursor = conn.cursor()
    cursor.execute(
        """
        UPDATE fraud_cases
        SET status=?, note=?, updated_at=?
        WHERE id=?
        """,
        (status, note, datetime.datetime.utcnow().isoformat(), case_id),
    )
    conn.commit()
    conn.close()


# ==========================================================
# Fraud Agent
# ==========================================================
class FraudAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="""
You are a calm, professional fraud-protection representative from a fictional bank called SecureBank.

Your goals:
1. Greet the customer warmly.
2. If user greets casually ("hi", "hello"), respond politely and ask for their full name.
3. After getting their name, load their fraud case from the database.
4. Verify identity using the security question from the case.
5. Describe the suspicious transaction calmly.
6. Ask if they made the transaction (yes/no).
7. If yes → mark case as confirmed_safe.
8. If no → mark case as confirmed_fraud.
9. If verification fails → mark as verification_failed and end call.
10. Close with a polite, reassuring message.

Never ask for PINs, full card numbers, passwords, or sensitive information.
"""
        )

        self.state = {
            "name": None,
            "case": None,
            "verified": False,
            "awaiting_security_answer": False,
            "awaiting_transaction_confirmation": False,
        }

    async def on_message(self, ctx, msg):
        text = msg.text.lower().strip()

        # -------------------------
        # 1. Greeting / Small Talk
        # -------------------------
        if self.state["name"] is None:
            if any(word in text for word in ["hi", "hello", "hey"]):
                await ctx.send_message(
                    "Hello! Thank you for speaking with SecureBank's Fraud Protection Team. "
                    "May I have your full name to look up your account?"
                )
                return

            # User gives name directly
            if "my name is" in text:
                name = text.replace("my name is", "").strip().title()
                self.state["name"] = name
            else:
                # We still need the name
                await ctx.send_message(
                    "To continue, may I have your full name, please?"
                )
                return

            # Load case from DB
            case = load_case_by_name(self.state["name"])
            if not case:
                await ctx.send_message(
                    f"I'm sorry, I couldn't find an account under the name {self.state['name']}. "
                    "Could you repeat your name?"
                )
                self.state["name"] = None
                return

            self.state["case"] = case
            self.state["awaiting_security_answer"] = True

            # Ask security question
            sec_question = case[3]
            await ctx.send_message(
                f"Thank you, {self.state['name']}. For verification, could you answer this question: {sec_question}?"
            )
            return

        # -------------------------
        # 2. Security Verification
        # -------------------------
        if self.state["awaiting_security_answer"]:
            expected = self.state["case"][4].lower()

            if expected in text:
                self.state["verified"] = True
                self.state["awaiting_security_answer"] = False
                self.state["awaiting_transaction_confirmation"] = True

                # Read suspicious transaction
                _, _, _, _, _, masked, amount, merchant, location, ttime, category, *_ = self.state["case"]

                await ctx.send_message(
                    f"Thank you. You're verified.\n"
                    f"We detected a suspicious transaction on your card ending {masked}.\n"
                    f"A charge of {amount} at {merchant} in {location} on {ttime}.\n"
                    f"Did you make this transaction?"
                )
                return

            else:
                # Verification failed
                update_case_status(self.state["case"][0], "verification_failed", "User failed security question.")
                await ctx.send_message(
                    "I'm sorry, that answer doesn't match our records. "
                    "For your safety, I cannot proceed further. Please contact SecureBank support."
                )
                await ctx.session.close()
                return

        # -------------------------
        # 3. Confirm or Deny Transaction
        # -------------------------
        if self.state["awaiting_transaction_confirmation"]:
            if "yes" in text:
                update_case_status(
                    self.state["case"][0],
                    "confirmed_safe",
                    "User confirmed the transaction as legitimate."
                )
                await ctx.send_message(
                    "Thank you for confirming. I have marked the transaction as safe. "
                    "No further action is needed. Have a great day!"
                )
                await ctx.session.close()
                return

            elif "no" in text:
                update_case_status(
                    self.state["case"][0],
                    "confirmed_fraud",
                    "User denied the transaction. Marked as fraud."
                )
                await ctx.send_message(
                    "Thank you for letting me know. I am marking this as fraudulent.\n"
                    "Your card will be temporarily blocked and our team will begin a dispute process.\n"
                    "We appreciate your quick response. Stay safe!"
                )
                await ctx.session.close()
                return

            else:
                await ctx.send_message("Please say 'yes' or 'no'. Did you make this transaction?")
                return


# ==========================================================
# LiveKit Setup
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

    usage = metrics.UsageCollector()
    @session.on("metrics_collected")
    def _on_metrics(ev: MetricsCollectedEvent):
        usage.collect(ev.metrics)

    await session.start(
        agent=FraudAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
