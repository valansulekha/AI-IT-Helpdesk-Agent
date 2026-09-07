import csv
import os
from datetime import datetime

from model import predict_issue
from knowledge_base import search_knowledge


TICKETS_FILE = "tickets.csv"


# ============================================================
# TOOL 1 — CLASSIFY ISSUE
# ============================================================

def classify_issue(user_message):
    """
    Uses the ML model to classify the user's IT problem.
    """

    category, confidence = predict_issue(user_message)

    return {
        "category": category,
        "confidence": round(float(confidence), 2)
    }


# ============================================================
# TOOL 2 — SEARCH KNOWLEDGE BASE (RAG)
# ============================================================

def search_knowledge_base(user_message, top_k=3):
    """
    Searches the RAG knowledge base for relevant IT solutions.
    """

    results = search_knowledge(user_message, top_k=top_k)

    return results


# ============================================================
# TOOL 3 — CREATE TICKET
# ============================================================

def create_ticket(issue, category, confidence, priority="Medium"):
    """
    Creates a support ticket and saves it to tickets.csv.
    """

    ticket_id = "TKT-" + datetime.now().strftime("%H%M%S")

    ticket = {
        "Ticket ID": ticket_id,
        "Issue": issue,
        "Category": category,
        "Confidence": f"{confidence:.2f}%",
        "Priority": priority,
        "Status": "Open",
        "Created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    file_exists = os.path.exists(TICKETS_FILE)

    with open(TICKETS_FILE, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=ticket.keys())

        if not file_exists or os.path.getsize(TICKETS_FILE) == 0:
            writer.writeheader()

        writer.writerow(ticket)

    return ticket


# ============================================================
# TOOL 4 — GET TICKET STATUS
# ============================================================

def get_ticket_status(ticket_id):
    """
    Finds the status of an existing ticket.
    """

    if not os.path.exists(TICKETS_FILE):
        return {
            "success": False,
            "message": "No tickets found."
        }

    with open(TICKETS_FILE, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            if row.get("Ticket ID") == ticket_id:
                return {
                    "success": True,
                    "ticket_id": ticket_id,
                    "status": row.get("Status"),
                    "category": row.get("Category"),
                    "priority": row.get("Priority")
                }

    return {
        "success": False,
        "message": f"Ticket {ticket_id} was not found."
    }


# ============================================================
# TOOL 5 — UPDATE TICKET STATUS
# ============================================================

def update_ticket_status(ticket_id, new_status):
    """
    Updates a ticket status.
    """

    valid_statuses = ["Open", "In Progress", "Resolved"]

    if new_status not in valid_statuses:
        return {
            "success": False,
            "message": "Invalid status."
        }

    if not os.path.exists(TICKETS_FILE):
        return {
            "success": False,
            "message": "No tickets found."
        }

    rows = []

    with open(TICKETS_FILE, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            if row.get("Ticket ID") == ticket_id:
                row["Status"] = new_status
            rows.append(row)

    found = any(row.get("Ticket ID") == ticket_id for row in rows)

    if not found:
        return {
            "success": False,
            "message": f"Ticket {ticket_id} was not found."
        }

    with open(TICKETS_FILE, "w", newline="", encoding="utf-8") as file:
        if rows:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    return {
        "success": True,
        "ticket_id": ticket_id,
        "new_status": new_status
    }


# ============================================================
# TOOL 6 — ESCALATE TO HUMAN
# ============================================================

def escalate_to_human(issue, reason):
    """
    Marks an issue for human IT-support intervention.
    """

    return {
        "escalated": True,
        "issue": issue,
        "reason": reason,
        "message": "This issue has been escalated to a human IT support agent."
    }


# ============================================================
# TOOL TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("AI IT HELPDESK TOOLS TEST")
    print("=" * 60)

    test_issue = "My laptop is very slow and keeps freezing"

    print("\n1. Testing classification tool...")
    classification = classify_issue(test_issue)
    print(classification)

    print("\n2. Testing RAG search tool...")
    results = search_knowledge_base(test_issue, top_k=2)

    for result in results:
        print(result)

    print("\n3. Testing escalation tool...")
    escalation = escalate_to_human(
        test_issue,
        "Low confidence prediction"
    )
    print(escalation)

    print("\n" + "=" * 60)
    print("TOOLS TEST COMPLETED")
    print("=" * 60)