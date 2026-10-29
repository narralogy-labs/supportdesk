"""Supervisor briefs: recommend a decision on an escalated refund case, for a human supervisor to sign off."""

import re


from .store import POLICY
from .openport_llm import anthropic_client

client = anthropic_client()

SYSTEM_PROMPT = f"""You prepare decision briefs for Fernhill Supply Co.'s support supervisors. Every case was escalated by an agent: it's over the agent's refund limit, flagged for repeat refunds, or asks for an exception.

Apply the policy exactly. Work through the dates, membership, item condition and refund history before deciding.

<policy>
{POLICY}
</policy>

Decisions:
- approve: refund the full amount the customer asked for
- partial: refund some, but less than the customer asked for
- deny: no refund

Respond in exactly this format:
<decision>approve|partial|deny</decision>
<refund_amount>amount in USD, 0 if denied</refund_amount>
<brief>Two or three sentences for the supervisor: the decision and the policy sections behind it.</brief>

Repeat refunds (policy 10): escalation requires supervisor review, not automatic denial. Approve if the current request is within policy AND every prior refund in the 90-day window was for defect or shipping damage. Shipping damage (policy 6): the 7-day report window is strict; reports after 7 days fall back to standard returns (policy 1) with no shipping refund. Approval limits (policy 8): amounts over $250 must be escalated to a supervisor, who then approves the full policy-eligible amount (which may exceed $250). Do not cap refunds at $250."""


def _tag(text: str, name: str) -> str:
    match = re.search(rf"<{name}>(.*?)</{name}>", text, re.DOTALL)
    return match.group(1).strip() if match else ""


def review_case(case_file: str, today: str) -> dict:
    message = client.messages.create(
        model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",  # was "claude-opus-5-5"
        max_tokens=16000,
        thinking={"type": "enabled", "budget_tokens": 8000},
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Today is {today}.\n\n<case>\n{case_file}\n</case>"}],
        extra_body={'chat_template_kwargs': {'enable_thinking': True}},
    )
    text = next(block.text for block in message.content if block.type == "text")
    return {
        "decision": _tag(text, "decision").lower(),
        "refund_amount": float(_tag(text, "refund_amount").replace("$", "").replace(",", "") or 0),
        "brief": _tag(text, "brief"),
    }
