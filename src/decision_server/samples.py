"""Bundled example requests shown in the UI.

Each sample is a ready-to-run :class:`DecisionRequest` payload plus the
metadata the UI needs to present it: a label, a category, a one-line summary
and the tasks it exercises. Samples use only generic content so they run on any
backend, and they are grouped so the UI can offer a browsable catalog instead of
a flat list.

Categories cover the three task types on their own, realistic multi-question
requests per domain (support, safety, commerce, finance, agents), and the corners
of the contract (optional fields, wide option sets, nested state, non-English
text).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .schema import DecisionRequest

CATEGORY_ORDER = [
    "Basics",
    "Support desk",
    "Trust & safety",
    "Commerce",
    "Finance",
    "Agents & RAG",
    "Contract corners",
]


@dataclass(frozen=True)
class Sample:
    """One bundled example: presentation metadata plus a request payload."""

    name: str
    label: str
    category: str
    summary: str
    payload: dict[str, Any]

    @property
    def tasks(self) -> list[str]:
        """Task types used by this sample, in question order, without repeats."""

        seen: list[str] = []
        for question in self.payload["questions"]:
            if question["task"] not in seen:
                seen.append(question["task"])
        return seen

    @property
    def question_count(self) -> int:
        return len(self.payload["questions"])

    def request(self) -> DecisionRequest:
        return DecisionRequest.model_validate(self.payload)

    def json(self, indent: int = 2) -> str:
        return json.dumps(self.payload, indent=indent, ensure_ascii=False)


# Sentences behind the batch example: the state and its questions stay in sync.
_BATCH_SENTENCES = [
    ("s1", "Water boils at 100 degrees Celsius at sea level."),
    ("s2", "The Eiffel Tower is located in Berlin."),
    ("s3", "Regular exercise tends to improve sleep quality."),
    ("s4", "Humans can breathe unaided underwater."),
    ("s5", "Adding salt to water raises its boiling point slightly."),
]

_SAMPLES: tuple[Sample, ...] = (
    # ── Basics: one question per task type ────────────────────────────────
    Sample(
        name="choice",
        label="Message triage",
        category="Basics",
        summary="A single choice question: pick the handling for one message.",
        payload={
            "state": {
                "message": "I've been feeling exhausted for three weeks and can't concentrate.",
            },
            "questions": [
                {
                    "id": "urgency",
                    "task": "choice",
                    "instructions": "Choose the most appropriate handling for this message.",
                    "options": [
                        {"id": "routine", "description": "Handle through the normal queue."},
                        {"id": "priority", "description": "Escalate to a priority review."},
                        {"id": "urgent", "description": "Treat as an urgent safety concern."},
                    ],
                }
            ],
        },
    ),
    Sample(
        name="noul",
        label="Yes / no judgment",
        category="Basics",
        summary="A single noul question: P(true) for one binary judgment.",
        payload={
            "state": {"sentence": "The library is closed on public holidays."},
            "questions": [
                {
                    "id": "entailment",
                    "task": "noul",
                    "instructions": "Is the statement supported by common sense reasoning?",
                    "options": [
                        {"id": "false", "description": "No"},
                        {"id": "true", "description": "Yes"},
                    ],
                }
            ],
        },
    ),
    Sample(
        name="score",
        label="Numeric scale",
        category="Basics",
        summary="A single score question: a probability over an ordered scale.",
        payload={
            "state": {"review": "Battery life is fine but the screen scratches easily."},
            "questions": [
                {
                    "id": "rating",
                    "task": "score",
                    "instructions": "Assign a numeric quality score from the criteria below.",
                    "options": [
                        {"id": "poor", "description": "Poor", "value": 1},
                        {"id": "fair", "description": "Fair", "value": 2},
                        {"id": "good", "description": "Good", "value": 3},
                        {"id": "excellent", "description": "Excellent", "value": 4},
                    ],
                }
            ],
        },
    ),
    Sample(
        name="mixed",
        label="Three tasks, one request",
        category="Basics",
        summary="choice, noul and score over the same state in one call.",
        payload={
            "state": {"text": "Vaccines are effective at preventing severe illness."},
            "questions": [
                {
                    "id": "agreement",
                    "task": "choice",
                    "instructions": "Pick the strongest stance this text expresses.",
                    "options": [
                        {"id": "support", "description": "Supports the claim."},
                        {"id": "neutral", "description": "Neither supports nor opposes."},
                        {"id": "oppose", "description": "Opposes the claim."},
                    ],
                },
                {
                    "id": "toxicity",
                    "task": "noul",
                    "instructions": "Does this text contain harassment?",
                    "options": [
                        {"id": "false", "description": "No"},
                        {"id": "true", "description": "Yes"},
                    ],
                },
                {
                    "id": "confidence",
                    "task": "score",
                    "instructions": "Rate how confident the text sounds.",
                    "options": [
                        {"id": "low", "description": "Low", "value": 1},
                        {"id": "mid", "description": "Medium", "value": 2},
                        {"id": "high", "description": "High", "value": 3},
                    ],
                },
            ],
        },
    ),
    # ── Support desk ──────────────────────────────────────────────────────
    Sample(
        name="ticket-routing",
        label="Route a ticket",
        category="Support desk",
        summary="Choice over teams, with tier and channel in the state.",
        payload={
            "state": {
                "ticket": {
                    "id": "TCK-4821",
                    "channel": "email",
                    "customer_tier": "business",
                    "subject": "Duplicate charge on invoice 2291",
                    "body": (
                        "We were billed twice for the same seat count on 3 March. "
                        "The second charge appeared two days later. Please refund one "
                        "of them and confirm the invoice is correct."
                    ),
                }
            },
            "questions": [
                {
                    "id": "team",
                    "task": "choice",
                    "instructions": "Which team should own this ticket?",
                    "options": [
                        {
                            "id": "billing",
                            "description": "Invoices, charges, refunds and payments.",
                        },
                        {"id": "shipping", "description": "Delivery, tracking and logistics."},
                        {
                            "id": "technical",
                            "description": "Bugs, outages, login and integrations.",
                        },
                        {"id": "account", "description": "Seats, plans, contracts and contacts."},
                    ],
                }
            ],
        },
    ),
    Sample(
        name="sla-breach",
        label="SLA breach risk",
        category="Support desk",
        summary="Noul over timestamps: will the first-response target be missed?",
        payload={
            "state": {
                "ticket": {
                    "priority": "high",
                    "opened_at": "2026-03-03T09:12:00Z",
                    "first_response_due": "2026-03-03T13:12:00Z",
                    "first_response_at": None,
                    "now": "2026-03-03T12:40:00Z",
                    "assignee_queue_depth": 34,
                }
            },
            "questions": [
                {
                    "id": "will_breach",
                    "task": "noul",
                    "instructions": "Will this ticket miss its first-response target?",
                    "options": [
                        {"id": "false", "description": "No, it will be answered in time."},
                        {"id": "true", "description": "Yes, it will breach the target."},
                    ],
                }
            ],
        },
    ),
    Sample(
        name="support-thread",
        label="Whole conversation",
        category="Support desk",
        summary="Routing, escalation and sentiment over a multi-turn thread.",
        payload={
            "state": {
                "account": {"id": "acct_77", "plan": "pro", "months_as_customer": 41},
                "thread": [
                    {
                        "from": "customer",
                        "text": "The export endpoint started returning 500s an hour ago.",
                    },
                    {"from": "agent", "text": "Thanks — can you share one request id?"},
                    {
                        "from": "customer",
                        "text": (
                            "req_9f2c. This is the third outage this month and my CFO is "
                            "asking questions."
                        ),
                    },
                    {
                        "from": "customer",
                        "text": "If it is not fixed today we will have to look at alternatives.",
                    },
                ],
            },
            "questions": [
                {
                    "id": "team",
                    "task": "choice",
                    "instructions": "Which team should handle this conversation next?",
                    "options": [
                        {"id": "technical", "description": "Engineering: bugs, outages, errors."},
                        {
                            "id": "account",
                            "description": "Account management: renewals, contracts.",
                        },
                        {"id": "billing", "description": "Billing: invoices and payments."},
                    ],
                },
                {
                    "id": "escalate",
                    "task": "noul",
                    "instructions": "Should this conversation be escalated to a manager?",
                    "options": [
                        {"id": "false", "description": "No, handle in the normal queue."},
                        {"id": "true", "description": "Yes, escalate now."},
                    ],
                },
                {
                    "id": "sentiment",
                    "task": "score",
                    "instructions": "Rate the customer's sentiment across the thread.",
                    "options": [
                        {"id": "very_negative", "description": "Very negative", "value": 1},
                        {"id": "negative", "description": "Negative", "value": 2},
                        {"id": "neutral", "description": "Neutral", "value": 3},
                        {"id": "positive", "description": "Positive", "value": 4},
                        {"id": "very_positive", "description": "Very positive", "value": 5},
                    ],
                },
            ],
        },
    ),
    Sample(
        name="csat-forecast",
        label="Predicted CSAT",
        category="Support desk",
        summary="Score a resolved conversation on a 1-5 satisfaction scale.",
        payload={
            "state": {
                "conversation": (
                    "Customer asked how to rotate an API key. The agent sent the steps, "
                    "the customer confirmed it worked after one reply, and the ticket was "
                    "closed in 22 minutes."
                ),
                "replies": 3,
                "handle_time_minutes": 22,
                "reopened": False,
            },
            "questions": [
                {
                    "id": "csat",
                    "task": "score",
                    "instructions": "Predict the satisfaction score the customer would give.",
                    "options": [
                        {"id": "1", "description": "Very unsatisfied", "value": 1},
                        {"id": "2", "description": "Unsatisfied", "value": 2},
                        {"id": "3", "description": "Neutral", "value": 3},
                        {"id": "4", "description": "Satisfied", "value": 4},
                        {"id": "5", "description": "Very satisfied", "value": 5},
                    ],
                }
            ],
        },
    ),
    # ── Trust & safety ────────────────────────────────────────────────────
    Sample(
        name="moderation-toxicity",
        label="Toxicity check",
        category="Trust & safety",
        summary="Noul gate on a user comment before it is published.",
        payload={
            "state": {
                "comment": "You clearly have no idea what you're talking about, delete this.",
                "author": {"account_age_days": 3, "previous_removals": 2},
                "surface": "public_forum",
            },
            "questions": [
                {
                    "id": "toxic",
                    "task": "noul",
                    "instructions": "Does this comment contain harassment or abuse?",
                    "options": [
                        {"id": "false", "description": "No, it is acceptable."},
                        {"id": "true", "description": "Yes, it should be held for review."},
                    ],
                }
            ],
        },
    ),
    Sample(
        name="policy-violation",
        label="Policy category",
        category="Trust & safety",
        summary="Choice over policy sections, including a 'no violation' option.",
        payload={
            "state": {
                "post": "Selling unused gift cards at a discount, DM me, no questions asked.",
                "context": "marketplace_listing",
            },
            "questions": [
                {
                    "id": "policy",
                    "task": "choice",
                    "instructions": "Which policy section does this post violate, if any?",
                    "options": [
                        {"id": "none", "description": "No violation."},
                        {
                            "id": "spam",
                            "description": "Unsolicited promotion or scam solicitation.",
                        },
                        {
                            "id": "prohibited_goods",
                            "description": "Selling items that may not be traded.",
                        },
                        {"id": "harassment", "description": "Targeted abuse of a person or group."},
                        {"id": "pii", "description": "Sharing private personal information."},
                    ],
                }
            ],
        },
    ),
    Sample(
        name="harm-severity",
        label="Severity grading",
        category="Trust & safety",
        summary="Score how severe a flagged item is on a 0-4 scale.",
        payload={
            "state": {
                "item": "A message telling someone they should hurt themselves.",
                "flagged_by": "user_report",
                "recipient_is_minor": False,
            },
            "questions": [
                {
                    "id": "severity",
                    "task": "score",
                    "instructions": "Grade the severity of this flagged item.",
                    "options": [
                        {"id": "none", "description": "No policy concern", "value": 0},
                        {"id": "low", "description": "Mild, low urgency", "value": 1},
                        {"id": "medium", "description": "Clear violation", "value": 2},
                        {"id": "high", "description": "Serious violation", "value": 3},
                        {"id": "critical", "description": "Immediate human review", "value": 4},
                    ],
                }
            ],
        },
    ),
    Sample(
        name="content-queue",
        label="Moderation queue",
        category="Trust & safety",
        summary="Three questions deciding whether to publish, hide or escalate.",
        payload={
            "state": {
                "item": {
                    "kind": "comment",
                    "text": "Here is the home address of the person who reported me, have fun.",
                    "author_warnings": 1,
                }
            },
            "questions": [
                {
                    "id": "action",
                    "task": "choice",
                    "instructions": "Choose the moderation action for this item.",
                    "options": [
                        {"id": "publish", "description": "Leave visible."},
                        {"id": "hide", "description": "Remove from public view."},
                        {"id": "escalate", "description": "Send to a human moderator immediately."},
                    ],
                },
                {
                    "id": "doxxing",
                    "task": "noul",
                    "instructions": "Does this item share private personal information?",
                    "options": [
                        {"id": "false", "description": "No"},
                        {"id": "true", "description": "Yes"},
                    ],
                },
                {
                    "id": "risk",
                    "task": "score",
                    "instructions": "Rate the risk of leaving this item visible.",
                    "options": [
                        {"id": "low", "description": "Low risk", "value": 1},
                        {"id": "medium", "description": "Medium risk", "value": 2},
                        {"id": "high", "description": "High risk", "value": 3},
                    ],
                },
            ],
        },
    ),
    # ── Commerce ──────────────────────────────────────────────────────────
    Sample(
        name="return-reason",
        label="Return reason",
        category="Commerce",
        summary="Choice over return reason codes from free-text plus order data.",
        payload={
            "state": {
                "order": {
                    "id": "ORD-10482",
                    "items": [{"sku": "SHOE-42-BLK", "name": "Trail shoes", "price": 89.0}],
                    "delivered_days_ago": 6,
                },
                "customer_message": "They arrived a size too small and the box was crushed.",
            },
            "questions": [
                {
                    "id": "reason",
                    "task": "choice",
                    "instructions": "Classify the primary reason for this return.",
                    "options": [
                        {"id": "wrong_size", "description": "Size or fit is wrong."},
                        {"id": "damaged", "description": "Item or packaging arrived damaged."},
                        {"id": "not_as_described", "description": "Differs from the listing."},
                        {"id": "changed_mind", "description": "No longer wanted."},
                    ],
                },
                {
                    "id": "reseller_fault",
                    "task": "noul",
                    "instructions": "Is the retailer at fault for this return?",
                    "options": [
                        {"id": "false", "description": "No"},
                        {"id": "true", "description": "Yes"},
                    ],
                },
            ],
        },
    ),
    Sample(
        name="chargeback-risk",
        label="Chargeback risk",
        category="Commerce",
        summary="Noul over transaction signals for a fraud queue.",
        payload={
            "state": {
                "transaction": {
                    "amount": 1240.0,
                    "currency": "USD",
                    "card_country": "NL",
                    "ship_country": "NG",
                    "is_new_account": True,
                    "velocity_last_hour": 4,
                    "device_matches_history": False,
                }
            },
            "questions": [
                {
                    "id": "review",
                    "task": "noul",
                    "instructions": "Should this transaction be held for manual review?",
                    "options": [
                        {"id": "false", "description": "No, let it through."},
                        {"id": "true", "description": "Yes, hold it."},
                    ],
                }
            ],
        },
    ),
    Sample(
        name="review-rating",
        label="Review stars",
        category="Commerce",
        summary="Score a product review on a 1-5 star scale.",
        payload={
            "state": {
                "review": {
                    "title": "Good, with one annoyance",
                    "body": (
                        "Cuts well and the battery lasts a week. The companion app asks me to "
                        "sign in every single time, which is irritating, but the device itself "
                        "is solid."
                    ),
                    "verified_purchase": True,
                }
            },
            "questions": [
                {
                    "id": "stars",
                    "task": "score",
                    "instructions": "Predict the star rating this review expresses.",
                    "options": [
                        {"id": "1", "description": "One star", "value": 1},
                        {"id": "2", "description": "Two stars", "value": 2},
                        {"id": "3", "description": "Three stars", "value": 3},
                        {"id": "4", "description": "Four stars", "value": 4},
                        {"id": "5", "description": "Five stars", "value": 5},
                    ],
                }
            ],
        },
    ),
    # ── Finance ───────────────────────────────────────────────────────────
    Sample(
        name="dispute-category",
        label="Card dispute type",
        category="Finance",
        summary="Choice over dispute reason codes from a claim narrative.",
        payload={
            "state": {
                "claim": (
                    "I never received the package but the tracking says delivered. "
                    "The merchant will not answer my emails."
                ),
                "amount": 210.5,
                "days_since_purchase": 18,
            },
            "questions": [
                {
                    "id": "reason_code",
                    "task": "choice",
                    "instructions": "Select the dispute reason code that fits this claim.",
                    "options": [
                        {"id": "not_received", "description": "Goods or services not received."},
                        {"id": "defective", "description": "Goods defective or not as described."},
                        {
                            "id": "unauthorised",
                            "description": "Transaction not recognised by the cardholder.",
                        },
                        {
                            "id": "duplicate",
                            "description": "Charged more than once for the same purchase.",
                        },
                        {"id": "cancelled", "description": "Recurring charge after cancellation."},
                    ],
                }
            ],
        },
    ),
    Sample(
        name="transaction-anomaly",
        label="Anomaly flag",
        category="Finance",
        summary="Noul against a customer's own spending baseline.",
        payload={
            "state": {
                "customer": {"monthly_spend_mean": 320.0, "monthly_spend_stddev": 60.0},
                "transaction": {
                    "amount": 2750.0,
                    "merchant_category": "jewellery",
                    "hour_local": 3,
                },
            },
            "questions": [
                {
                    "id": "anomalous",
                    "task": "noul",
                    "instructions": "Is this transaction anomalous for this customer?",
                    "options": [
                        {"id": "false", "description": "No, it looks typical."},
                        {"id": "true", "description": "Yes, it deviates from the baseline."},
                    ],
                }
            ],
        },
    ),
    Sample(
        name="credit-band",
        label="Credit band",
        category="Finance",
        summary="Score an application onto ordered risk bands.",
        payload={
            "state": {
                "application": {
                    "requested_amount": 4500,
                    "months_employed": 14,
                    "missed_payments_12m": 1,
                    "utilisation_pct": 78,
                }
            },
            "questions": [
                {
                    "id": "band",
                    "task": "score",
                    "instructions": "Assign the risk band for this application.",
                    "options": [
                        {"id": "A", "description": "Lowest risk", "value": 1},
                        {"id": "B", "description": "Below average risk", "value": 2},
                        {"id": "C", "description": "Average risk", "value": 3},
                        {"id": "D", "description": "Above average risk", "value": 4},
                        {"id": "E", "description": "Highest risk", "value": 5},
                    ],
                }
            ],
        },
    ),
    # ── Agents & RAG ──────────────────────────────────────────────────────
    Sample(
        name="tool-selection",
        label="Tool selection",
        category="Agents & RAG",
        summary="Choice over callable tools described with their arguments.",
        payload={
            "state": {
                "user_turn": "What's the weather in Lisbon tomorrow morning?",
                "tools": [
                    {"name": "get_forecast", "args": ["location", "date"]},
                    {"name": "search_web", "args": ["query"]},
                    {"name": "send_email", "args": ["to", "body"]},
                ],
            },
            "questions": [
                {
                    "id": "tool",
                    "task": "choice",
                    "instructions": "Which tool should the agent call next?",
                    "options": [
                        {
                            "id": "get_forecast",
                            "description": (
                                "get_forecast(location, date) — structured weather data "
                                "for a place and day."
                            ),
                        },
                        {
                            "id": "search_web",
                            "description": "search_web(query) — free-text web search.",
                        },
                        {
                            "id": "send_email",
                            "description": "send_email(to, body) — deliver a message to a person.",
                        },
                        {
                            "id": "none",
                            "description": "Answer directly without calling a tool.",
                        },
                    ],
                }
            ],
        },
    ),
    Sample(
        name="grounded-answer",
        label="Grounded answer",
        category="Agents & RAG",
        summary="Noul: is the draft answer supported by the retrieved context?",
        payload={
            "state": {
                "question": "How long do I have to return an item?",
                "context": [
                    "Returns are accepted within 30 days of delivery.",
                    "Downloaded software cannot be returned.",
                ],
                "draft_answer": "You have 30 days from delivery to return most items.",
            },
            "questions": [
                {
                    "id": "grounded",
                    "task": "noul",
                    "instructions": "Is the draft answer fully supported by the context?",
                    "options": [
                        {"id": "false", "description": "No, it adds or contradicts the context."},
                        {"id": "true", "description": "Yes, it is entailed by the context."},
                    ],
                }
            ],
        },
    ),
    Sample(
        name="passage-rerank",
        label="Passage reranking",
        category="Agents & RAG",
        summary="Choice across six retrieved passages for one query.",
        payload={
            "state": {
                "query": "How do I reset a forgotten password?",
                "retriever": "bm25",
                "candidate_ids": ["p1", "p2", "p3", "p4", "p5", "p6"],
            },
            "questions": [
                {
                    "id": "best_passage",
                    "task": "choice",
                    "instructions": "Which retrieved passage best answers the query?",
                    "options": [
                        {
                            "id": "p1",
                            "description": "Password reset links expire after 30 minutes.",
                        },
                        {
                            "id": "p2",
                            "description": (
                                "Choose Reset password on the sign-in page to receive an "
                                "email link."
                            ),
                        },
                        {"id": "p3", "description": "Two-factor codes are six digits long."},
                        {
                            "id": "p4",
                            "description": (
                                "Administrators can force a reset for any member of the workspace."
                            ),
                        },
                        {
                            "id": "p5",
                            "description": "Billing questions are handled by the finance team.",
                        },
                        {
                            "id": "p6",
                            "description": "Passwords must be at least twelve characters.",
                        },
                    ],
                }
            ],
        },
    ),
    Sample(
        name="agent-loop",
        label="Agent control loop",
        category="Agents & RAG",
        summary="Next action, stop decision and confidence for one agent step.",
        payload={
            "state": {
                "goal": "Book the cheapest direct flight from Berlin to Oslo on 12 May.",
                "steps_taken": 3,
                "last_tool_result": {
                    "tool": "search_flights",
                    "found": [
                        {"price": 84, "direct": True, "depart": "06:40"},
                        {"price": 71, "direct": False, "depart": "13:05"},
                    ],
                },
                "scratchpad": (
                    "Direct options exist; the cheaper one has a stopover, so the goal is "
                    "met by the 84 EUR flight."
                ),
            },
            "questions": [
                {
                    "id": "next_action",
                    "task": "choice",
                    "instructions": "What should the agent do next?",
                    "options": [
                        {
                            "id": "book",
                            "description": "Call book_flight with the best direct option.",
                        },
                        {"id": "search_again", "description": "Broaden the flight search."},
                        {"id": "ask_user", "description": "Ask the user to confirm the choice."},
                        {"id": "finish", "description": "Stop and report the result."},
                    ],
                },
                {
                    "id": "goal_met",
                    "task": "noul",
                    "instructions": "Has enough information been gathered to meet the goal?",
                    "options": [
                        {"id": "false", "description": "No, more steps are needed."},
                        {"id": "true", "description": "Yes, the goal can be completed now."},
                    ],
                },
                {
                    "id": "confidence",
                    "task": "score",
                    "instructions": "Rate confidence in the chosen next action.",
                    "options": [
                        {"id": "low", "description": "Low", "value": 1},
                        {"id": "medium", "description": "Medium", "value": 2},
                        {"id": "high", "description": "High", "value": 3},
                    ],
                },
            ],
        },
    ),
    # ── Contract corners ──────────────────────────────────────────────────
    Sample(
        name="minimal",
        label="Minimal request",
        category="Contract corners",
        summary="Smallest valid request: a string state and two options.",
        payload={
            "state": "The parcel arrived two days late.",
            "questions": [
                {
                    "id": "negative",
                    "task": "noul",
                    "instructions": "Is this statement negative?",
                    "options": [
                        {"id": "false", "description": "No"},
                        {"id": "true", "description": "Yes"},
                    ],
                }
            ],
        },
    ),
    Sample(
        name="wide-choice",
        label="Ten-way choice",
        category="Contract corners",
        summary="A wide option set, as used for intent classification.",
        payload={
            "state": {"utterance": "Can I move my subscription to the team plan next month?"},
            "questions": [
                {
                    "id": "intent",
                    "task": "choice",
                    "instructions": "Classify the intent of this utterance.",
                    "options": [
                        {
                            "id": "upgrade_plan",
                            "description": "Change to a different plan or tier.",
                        },
                        {"id": "cancel", "description": "End the subscription."},
                        {
                            "id": "billing_question",
                            "description": "Ask about an invoice or payment.",
                        },
                        {"id": "refund", "description": "Request money back."},
                        {"id": "technical_issue", "description": "Report a bug or error."},
                        {"id": "how_to", "description": "Ask how a feature works."},
                        {"id": "account_access", "description": "Login, password or seat access."},
                        {"id": "complaint", "description": "Express dissatisfaction."},
                        {"id": "praise", "description": "Express satisfaction."},
                        {"id": "other", "description": "None of the above."},
                    ],
                }
            ],
        },
    ),
    Sample(
        name="optional-fields",
        label="Optional fields",
        category="Contract corners",
        summary="system_prompt, state_json, instructions_json and description_json.",
        payload={
            "state": {"temperature_c": -4, "road_state": "wet"},
            "state_json": '{"temperature_c": -4, "road_state": "wet"}',
            "questions": [
                {
                    "id": "gritting",
                    "task": "choice",
                    "instructions": "Should gritters be dispatched on this route?",
                    "instructions_json": (
                        '{"decision": "dispatch gritters", "basis": "sensor readings"}'
                    ),
                    "system_prompt": (
                        "You are a winter road maintenance controller. Answer from the "
                        "sensor state only."
                    ),
                    "options": [
                        {
                            "id": "dispatch",
                            "description": "Send gritters now.",
                            "description_json": '{"action": "dispatch", "window_minutes": 90}',
                        },
                        {
                            "id": "monitor",
                            "description": "Keep monitoring the sensors.",
                            "description_json": '{"action": "monitor", "window_minutes": 240}',
                        },
                        {
                            "id": "stand_down",
                            "description": "No action needed.",
                            "description_json": '{"action": "none"}',
                        },
                    ],
                }
            ],
        },
    ),
    Sample(
        name="nested-state",
        label="Deeply nested state",
        category="Contract corners",
        summary="State can be any JSON: nested objects and arrays are passed through.",
        payload={
            "state": {
                "order": {
                    "id": "ORD-9931",
                    "customer": {
                        "id": "cust_4",
                        "segment": "enterprise",
                        "history": {"orders": 62, "disputes": 1, "avg_basket": 1480.0},
                    },
                    "lines": [
                        {"sku": "A-1", "qty": 40, "unit_price": 12.0, "backordered": True},
                        {"sku": "B-7", "qty": 5, "unit_price": 210.0, "backordered": False},
                    ],
                    "shipping": {"method": "express", "promised_days": 2, "delayed_days": 4},
                    "payment": {"method": "invoice", "terms_days": 30, "overdue": False},
                }
            },
            "questions": [
                {
                    "id": "intervention",
                    "task": "choice",
                    "instructions": "Choose the intervention for this order.",
                    "options": [
                        {"id": "none", "description": "Let it proceed unchanged."},
                        {"id": "notify", "description": "Notify the customer about the delay."},
                        {"id": "expedite", "description": "Expedite shipping at our cost."},
                        {"id": "manual_review", "description": "Route to an account manager."},
                    ],
                }
            ],
        },
    ),
    Sample(
        name="multilingual",
        label="Non-English state",
        category="Contract corners",
        summary="Instructions in English, state in Japanese, Thai and Spanish.",
        payload={
            "state": {
                "messages": [
                    {
                        "lang": "ja",
                        "text": "注文した商品がまだ届いていません。確認していただけますか。",
                    },
                    {"lang": "th", "text": "สินค้าเสียหาย กรุณาคืนเงินด้วยครับ"},
                    {"lang": "es", "text": "¡Excelente servicio, muchas gracias por la rapidez!"},
                ]
            },
            "questions": [
                {
                    "id": "needs_agent",
                    "task": "noul",
                    "instructions": "Does at least one of these messages need a human agent?",
                    "options": [
                        {"id": "false", "description": "No"},
                        {"id": "true", "description": "Yes"},
                    ],
                },
                {
                    "id": "overall_tone",
                    "task": "score",
                    "instructions": "Rate the overall tone of the three messages together.",
                    "options": [
                        {"id": "negative", "description": "Mostly negative", "value": -1},
                        {"id": "neutral", "description": "Mixed or neutral", "value": 0},
                        {"id": "positive", "description": "Mostly positive", "value": 1},
                    ],
                },
            ],
        },
    ),
    Sample(
        name="fractional-scale",
        label="Fractional scale",
        category="Contract corners",
        summary="Score values may be negative or fractional; they need not be integers.",
        payload={
            "state": {
                "headline": "Study hints the new drug could slow the disease, researchers caution.",
            },
            "questions": [
                {
                    "id": "stance",
                    "task": "score",
                    "instructions": (
                        "Place the headline on a stance scale from opposing to supporting "
                        "the claim."
                    ),
                    "options": [
                        {
                            "id": "strongly_against",
                            "description": "Strongly against",
                            "value": -1.0,
                        },
                        {"id": "against", "description": "Against", "value": -0.5},
                        {"id": "neutral", "description": "Neutral", "value": 0.0},
                        {"id": "for", "description": "For", "value": 0.5},
                        {"id": "strongly_for", "description": "Strongly for", "value": 1.0},
                    ],
                }
            ],
        },
    ),
    Sample(
        name="batch-noul",
        label="Batch of judgments",
        category="Contract corners",
        summary="Many questions of one type in a single request.",
        payload={
            "state": {"sentences": [{"id": sid, "text": text} for sid, text in _BATCH_SENTENCES]},
            "questions": [
                {
                    "id": sid,
                    "task": "noul",
                    "instructions": f"Is sentence {sid} factually correct?",
                    "options": [
                        {"id": "false", "description": "No"},
                        {"id": "true", "description": "Yes"},
                    ],
                }
                for sid, _ in _BATCH_SENTENCES
            ],
        },
    ),
)

_BY_NAME: dict[str, Sample] = {s.name: s for s in _SAMPLES}


def samples() -> list[Sample]:
    """Every bundled example, in catalog order."""

    return list(_SAMPLES)


def categories() -> list[str]:
    """Category names in display order, limited to categories that exist."""

    present = {s.category for s in _SAMPLES}
    return [c for c in CATEGORY_ORDER if c in present]


def sample(name: str) -> Sample:
    """One example with its metadata; raises ``KeyError`` if unknown."""

    if name not in _BY_NAME:
        raise KeyError(name)
    return _BY_NAME[name]


def sample_names() -> list[str]:
    return [s.name for s in _SAMPLES]


def sample_request(name: str) -> DecisionRequest:
    return sample(name).request()


def sample_json(name: str, indent: int = 2) -> str:
    return sample(name).json(indent=indent)


__all__ = [
    "CATEGORY_ORDER",
    "Sample",
    "categories",
    "sample",
    "sample_json",
    "sample_names",
    "sample_request",
    "samples",
]
