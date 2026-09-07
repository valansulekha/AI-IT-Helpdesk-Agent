from tools import (
    classify_issue,
    search_knowledge_base,
    create_ticket,
    get_ticket_status,
    update_ticket_status,
    escalate_to_human
)


# ============================================================
# AI HELPDESK AGENT
# ============================================================

class HelpdeskAgent:

    def __init__(self):
        self.name = "AI IT Helpdesk Agent"

    # --------------------------------------------------------
    # DETERMINE PRIORITY
    # --------------------------------------------------------

    def determine_priority(self, category):

        if category == "Security Issue":
            return "High"

        elif category in [
            "Network Issue",
            "Printer Issue",
            "Account Issue",
            "Hardware Issue",
            "Database Issue"
        ]:
            return "Medium"

        else:
            return "Low"

    # --------------------------------------------------------
    # DECIDE WHETHER HUMAN REVIEW IS REQUIRED
    # --------------------------------------------------------

    def needs_human_review(self, confidence, category):

        if category == "Security Issue":
            return True

        if confidence < 60:
            return True

        return False

    # --------------------------------------------------------
    # MAIN AGENT
    # --------------------------------------------------------

    def process_request(self, user_message):

        print("\n" + "=" * 60)
        print("AI HELPDESK AGENT")
        print("=" * 60)

        print("\nUser Request:")
        print(user_message)

        # ====================================================
        # STEP 1 — CLASSIFY USING ML TOOL
        # ====================================================

        print("\n[AGENT] Calling classification tool...")

        classification = classify_issue(user_message)

        category = classification["category"]
        confidence = classification["confidence"]

        print(f"[AGENT] Category: {category}")
        print(f"[AGENT] Confidence: {confidence:.2f}%")

        # ====================================================
        # STEP 2 — SEARCH RAG KNOWLEDGE BASE
        # ====================================================

        print("\n[AGENT] Searching knowledge base...")

        knowledge_results = search_knowledge_base(
            user_message,
            top_k=3
        )

        # ====================================================
        # CHECK WHETHER KNOWLEDGE WAS FOUND
        # ====================================================

        if knowledge_results:

            best_result = knowledge_results[0]

            similarity = best_result.get(
                "similarity_score",
                0
            )

            print(
                f"[AGENT] Best knowledge match: "
                f"{best_result['title']}"
            )

            print(
                f"[AGENT] Similarity: "
                f"{similarity:.3f}"
            )

        else:

            best_result = None
            similarity = 0

            print("[AGENT] No knowledge match found.")

        # ====================================================
        # STEP 3 — DETERMINE PRIORITY
        # ====================================================

        priority = self.determine_priority(
            category
        )

        print(
            f"\n[AGENT] Priority: {priority}"
        )

        # ====================================================
        # STEP 4 — HUMAN REVIEW DECISION
        # ====================================================

        human_review = self.needs_human_review(
            confidence,
            category
        )

        if human_review:

            print(
                "[AGENT] Human review is recommended."
            )

        else:

            print(
                "[AGENT] Human review is not required."
            )

        # ====================================================
        # STEP 5 — GENERATE SOLUTION
        # ====================================================

        if best_result:

            solution = best_result["content"].strip()

        else:

            solution = (
                "No matching solution was found in "
                "the knowledge base."
            )

        # ====================================================
        # STEP 6 — CREATE TICKET
        # ====================================================

        print("\n[AGENT] Creating support ticket...")

        ticket = create_ticket(
            user_message,
            category,
            confidence,
            priority
        )

        print(
            f"[AGENT] Ticket created: "
            f"{ticket['Ticket ID']}"
        )

        # ====================================================
        # STEP 7 — ESCALATE IF REQUIRED
        # ====================================================

        escalation = None

        if human_review:

            escalation = escalate_to_human(
                user_message,
                (
                    f"Category: {category}; "
                    f"Confidence: {confidence:.2f}%"
                )
            )

            print(
                "[AGENT] Issue escalated to human support."
            )

        # ====================================================
        # FINAL RESPONSE
        # ====================================================

        response = {

            "agent": self.name,

            "issue": user_message,

            "category": category,

            "confidence": confidence,

            "priority": priority,

            "human_review": human_review,

            "knowledge_title": (
                best_result["title"]
                if best_result
                else None
            ),

            "knowledge_similarity": similarity,

            "solution": solution,

            "ticket": ticket,

            "escalation": escalation
        }

        print("\n" + "=" * 60)
        print("AGENT RESPONSE")
        print("=" * 60)

        print(
            f"\nCategory: {category}"
        )

        print(
            f"Confidence: {confidence:.2f}%"
        )

        print(
            f"Priority: {priority}"
        )

        print(
            f"\nSolution:\n{solution}"
        )

        print(
            f"\nTicket ID: "
            f"{ticket['Ticket ID']}"
        )

        if human_review:

            print(
                "\n⚠ Human Review: REQUIRED"
            )

        else:

            print(
                "\n✓ Human Review: NOT REQUIRED"
            )

        print("\n" + "=" * 60)

        return response


# ============================================================
# AGENT TEST
# ============================================================

if __name__ == "__main__":

    agent = HelpdeskAgent()

    test_requests = [

        "My laptop is very slow and keeps freezing",

        "My Wi-Fi is not connecting",

        "I received a suspicious phishing email"

    ]

    for request in test_requests:

        agent.process_request(request)

        print("\n")