"""Ticket triage: route each incoming ticket to a queue and set its priority."""

import json

from .openport_llm import token_factory_client

client = token_factory_client()

SYSTEM_PROMPT = """You are the triage system for Fernhill Supply Co., an online outdoor gear store.
Classify the customer's ticket.

Categories:
- billing: charges, invoices, payment methods, promo codes, price adjustments
- shipping: delivery status, delays, lost or misrouted packages, address changes before delivery
- returns: returning, exchanging or refunding an item, including damaged or defective items
- product_question: sizing, specs, compatibility, care instructions, stock questions
- account: login, password, email changes, membership, deleting an account
- technical: website or app errors, checkout bugs
- other: anything else, including spam and praise

Priority:
- urgent: safety risks (injury, fire, gas leaks, gear failing in the field), legal threats, or a customer who is stranded or at risk
- normal: most tickets
- low: general questions with no time pressure, praise, spam

If a ticket raises several issues, pick the category of the one the customer most needs resolved.
Respond with JSON only: {"category": "<category>", "priority": "<priority>"}

Priority rules: General product questions (sizing, care, specs, compatibility) = low. Legal threats = urgent. Safety risks (injury, stranded, gear failing in field) = urgent. Category rules: Defective/damaged items causing safety issues = returns. Refund/exchange/return requests = returns. Apply the customer's most pressing need."""


def triage_ticket(ticket: str) -> dict:
    response = client.chat.completions.create(
        model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",  # was "gpt-5-mini"
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ticket},
        ],
        response_format={"type": "json_object"},
        extra_body={'chat_template_kwargs': {'enable_thinking': True}},
    )
    return json.loads(response.choices[0].message.content)
