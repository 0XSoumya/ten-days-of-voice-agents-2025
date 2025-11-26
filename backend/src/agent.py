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


# ----------------------
# Utilities / Data Load
# ----------------------
FAQ_PATH = "shared-data/day5_faq_instamart.json"
LEADS_DIR = "leads"
LEADS_INDEX = os.path.join(LEADS_DIR, "leads.json")


def ensure_leads_dir():
    os.makedirs(LEADS_DIR, exist_ok=True)
    if not os.path.exists(LEADS_INDEX):
        with open(LEADS_INDEX, "w") as f:
            json.dump([], f, indent=2)


def load_faq():
    """Load FAQ JSON (list of entries with id, keywords, question, answer)."""
    if not os.path.exists(FAQ_PATH):
        logger.error("FAQ file not found at %s", FAQ_PATH)
        return []
    with open(FAQ_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def keyword_match(faq_list, user_text):
    """Simple keyword-based match: returns the best matching FAQ answer or None."""
    user_text_lower = user_text.lower()
    best = None
    best_count = 0
    for item in faq_list:
        kws = item.get("keywords", []) or []
        count = 0
        for kw in kws:
            if kw.lower() in user_text_lower:
                count += 1
        if count > best_count:
            best_count = count
            best = item
    # require at least one keyword match
    return best if best_count > 0 else None


# -------------------------
# SDR Agent Implementation
# -------------------------
class SDRAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
You are an SDR (Sales Development Representative) voice assistant for Instamart (Swiggy).
Behave warmly and professionally. Your goals:
- Greet the visitor, ask what brought them here, and understand their needs.
- Answer product/pricing/support questions using ONLY the provided Instamart FAQ dataset.
- If the user is interested (requests sign-up, demo, pricing, contact), collect lead details naturally.
- When the user says they are done, give a short summary of collected lead data and save it.
Be concise and polite. Never invent product facts not in the FAQ — if you don't know, offer to capture the lead and follow up.
"""
        )

        # state for mention of FAQ, lead capture, etc.
        self.faq = load_faq()
        # Lead capture state template
        self.lead_template = {
            "name": None,
            "company": None,
            "email": None,
            "role": None,
            "use_case": None,
            "team_size": None,
            "timeline": None,
            "created_at": None,
        }
        # per-session state
        self.session_state = {
            "collecting_lead": False,
            "lead": None,
            "last_faq_hit": None,
        }

    # ---------------------
    # Tool: save lead to disk
    # ---------------------
    @function_tool
    async def save_lead(self, ctx: RunContext, lead: dict):
        """Save a lead dict to a timestamped JSON and append to an index file."""
        ensure_leads_dir()
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        filename = os.path.join(LEADS_DIR, f"lead_{ts}.json")
        lead_copy = dict(lead)
        # ensure created_at
        if not lead_copy.get("created_at"):
            lead_copy["created_at"] = datetime.now().isoformat()
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(lead_copy, f, indent=2)
        # append to index
        with open(LEADS_INDEX, "r+", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except Exception:
                data = []
            data.append({"file": filename, "name": lead_copy.get("name"), "email": lead_copy.get("email"), "created_at": lead_copy["created_at"]})
            f.seek(0)
            json.dump(data, f, indent=2)
            f.truncate()
        return f"Lead saved to {filename}"

    # ---------------------
    # Helper to get next missing lead field
    # ---------------------
    def next_missing_lead_field(self):
        lead = self.session_state.get("lead") or {}
        for k, v in lead.items():
            if v in (None, ""):
                return k
        return None

    # ---------------------
    # Start lead capture flow
    # ---------------------
    def start_lead_capture(self):
        self.session_state["collecting_lead"] = True
        lead = dict(self.lead_template)
        lead["created_at"] = datetime.now().isoformat()
        self.session_state["lead"] = lead

    # ---------------------
    # Main message handler
    # ---------------------
    async def on_message(self, ctx, msg):
        text = (msg.text or "").strip()
        if not text:
            return

        text_l = text.lower()

        # 1) End-of-call detection
        if any(phrase in text_l for phrase in ["that's all", "i'm done", "i am done", "thanks", "thank you", "that is all"]):
            # If collecting lead, finalize and save
            if self.session_state.get("collecting_lead") and self.session_state.get("lead"):
                lead = self.session_state["lead"]
                # finalize summary
                summary = self._lead_summary(lead)
                # save via tool (ensures persistence)
                await ctx.call_tool(self.save_lead, lead=lead)
                # reset collecting flag
                self.session_state["collecting_lead"] = False
                self.session_state["lead"] = None
                await ctx.send_message(f"Thanks — {summary} I have saved your details and someone will follow up.")
            else:
                await ctx.send_message("Thanks for your time — have a great day!")
            return

        # 2) If currently collecting lead, ask for missing fields sequentially
        if self.session_state.get("collecting_lead"):
            lead = self.session_state.get("lead")
            # decide which field we are waiting for
            next_field = self.next_missing_lead_field()
            # Try to heuristically fill based on user reply
            if next_field:
                filled = False
                # simple heuristics by field
                if next_field == "name":
                    # Assume the user reply is their name
                    lead["name"] = text.strip()
                    filled = True
                elif next_field == "company":
                    lead["company"] = text.strip()
                    filled = True
                elif next_field == "email":
                    if "@" in text and "." in text:
                        lead["email"] = text.strip()
                        filled = True
                    else:
                        # ask again if no email-like pattern
                        await ctx.send_message("Could you share the best email to reach you at?")
                        return
                elif next_field == "role":
                    lead["role"] = text.strip()
                    filled = True
                elif next_field == "use_case":
                    lead["use_case"] = text.strip()
                    filled = True
                elif next_field == "team_size":
                    lead["team_size"] = text.strip()
                    filled = True
                elif next_field == "timeline":
                    lead["timeline"] = text.strip()
                    filled = True

                if filled:
                    # If there are more fields, ask next
                    nm = self.next_missing_lead_field()
                    if nm:
                        prompt = self._lead_prompt_for_field(nm)
                        await ctx.send_message(prompt)
                    else:
                        # All fields collected — summarize and save
                        summary = self._lead_summary(lead)
                        await ctx.call_tool(self.save_lead, lead=lead)
                        # reset
                        self.session_state["collecting_lead"] = False
                        self.session_state["lead"] = None
                        await ctx.send_message(f"Perfect — {summary} I have saved your details and someone from the team will reach out soon.")
                return

        # 3) FAQ matching (only answer from FAQ content)
        faq_hit = keyword_match(self.faq, text)
        if faq_hit:
            self.session_state["last_faq_hit"] = faq_hit["id"]
            await ctx.send_message(faq_hit["answer"])
            # after answering, check if user signals interest
            if any(word in text_l for word in ["interested", "sign up", "signup", "demo", "talk to sales", "pricing", "get in touch", "contact"]):
                # start lead capture
                if not self.session_state.get("collecting_lead"):
                    self.start_lead_capture()
                    await ctx.send_message("Great — I can capture a few details to help our team follow up. What's your name?")
            return

        # 4) If user expresses interest directly (no FAQ hit) — start lead capture
        if any(word in text_l for word in ["i'm interested", "i am interested", "sign me up", "i want to sign up", "i want a demo", "contact me", "get in touch", "i want to speak to sales"]):
            if not self.session_state.get("collecting_lead"):
                self.start_lead_capture()
                await ctx.send_message("Fantastic — I can capture a few details to pass to our team. What's your name?")
            else:
                # already collecting - ask next field
                nm = self.next_missing_lead_field()
                if nm:
                    await ctx.send_message(self._lead_prompt_for_field(nm))
            return

        # 5) If the message looks like a greeting, respond and prompt
        if any(g in text_l for g in ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]):
            await ctx.send_message("Hi! Welcome to Instamart — I’m the Instamart assistant. What brought you here today? (questions about orders, delivery, refunds, or interested in a partnership/demo?)")
            return

        # 6) If none of the above, fallback — ask clarifying question and offer lead capture
        await ctx.send_message(
            "I’m sorry — I don't have a direct answer to that. I can capture your contact and have someone follow up, or you can ask another Instamart-related question."
        )
        # Offer to capture lead
        if any(word in text_l for word in ["follow up", "call me", "email me", "contact", "reach out"]):
            if not self.session_state.get("collecting_lead"):
                self.start_lead_capture()
                await ctx.send_message("Sure — I can take your details. What's your name?")


    # ---------------------
    # Small helpers
    # ---------------------
    def _lead_prompt_for_field(self, field):
        prompts = {
            "name": "What's your full name?",
            "company": "Which company are you from (if any)?",
            "email": "What's the best email to reach you at?",
            "role": "What's your role there?",
            "use_case": "Briefly, what would you like to use Instamart for?",
            "team_size": "How large is your team (approx)?",
            "timeline": "What's your timeline — now, soon, or later?"
        }
        return prompts.get(field, f"Please provide {field}.")

    def _lead_summary(self, lead):
        # Build a short human-readable summary
        parts = []
        if lead.get("name"):
            parts.append(lead["name"])
        if lead.get("role"):
            parts.append(f"({lead['role']})")
        if lead.get("company"):
            parts.append(f"from {lead['company']}")
        identity = " ".join(parts).strip()
        needs = lead.get("use_case", "no specific use case provided")
        team = f"Team size: {lead['team_size']}" if lead.get("team_size") else ""
        timeline = f"Timeline: {lead['timeline']}" if lead.get("timeline") else ""
        summary = f"{identity} — interested in: {needs}. {team} {timeline}".strip()
        # tidy spaces
        summary = " ".join(summary.split())
        return summary


# -------------------------
# LiveKit session bootstrap
# -------------------------
def prewarm(proc: JobProcess):
    # Load VAD model in advance to reduce cold-start latency
    proc.userdata["vad"] = silero.VAD.load()
    # Preload FAQ into agent? (agent loads on init)
    logger.info("Prewarmed VAD and resources.")


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

    # metrics collector
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _metrics(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    # start the agent session
    await session.start(
        agent=SDRAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # connect the agent
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
