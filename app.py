import streamlit as st
import pandas as pd
import os

from agent import HelpdeskAgent
from model import model, vectorizer
from tools import update_ticket_status


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI IT Helpdesk Agent",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🤖 AI IT Helpdesk Agent")

st.markdown(
    """
    **Intelligent IT support using AI Agent + Machine Learning + RAG + Tools**
    
    The system analyzes the user's IT problem, predicts the issue category,
    retrieves relevant knowledge, provides a solution, creates a support ticket,
    and escalates critical issues when required.
    """
)

st.divider()


# ============================================================
# CREATE AGENT
# ============================================================

agent = HelpdeskAgent()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ System Components")

st.sidebar.success("✅ Machine Learning")
st.sidebar.success("✅ AI Agent")
st.sidebar.success("✅ RAG Knowledge Base")
st.sidebar.success("✅ Tool Calling")
st.sidebar.success("✅ Explainable AI")
st.sidebar.success("✅ Ticket Management")

st.sidebar.divider()

st.sidebar.info(
    """
    **Agent Workflow**

    User Issue
    ↓
    Issue Classification
    ↓
    RAG Knowledge Search
    ↓
    Priority Decision
    ↓
    Solution Generation
    ↓
    Ticket Creation
    ↓
    Human Escalation if Required
    """
)


# ============================================================
# MAIN USER INPUT
# ============================================================

st.header("📝 Submit IT Issue")

user_message = st.text_area(
    "Describe your IT problem:",
    placeholder=(
        "Example: My laptop is very slow and keeps freezing..."
    ),
    height=120
)


# ============================================================
# PROCESS REQUEST
# ============================================================

if st.button("🚀 Ask AI Helpdesk", type="primary"):

    if not user_message.strip():

        st.warning("⚠️ Please enter an IT problem first.")

    else:

        with st.spinner("🤖 AI Helpdesk Agent is analyzing your issue..."):

            result = agent.process_request(user_message)

        st.success("✅ AI Agent completed the analysis.")

        st.divider()


        # ====================================================
        # GET AGENT RESULTS
        # ====================================================

        category = result.get(
            "category",
            "Unknown"
        )

        confidence = result.get(
            "confidence",
            0
        )

        priority = result.get(
            "priority",
            "Medium"
        )

        human_review = result.get(
            "human_review",
            False
        )

        knowledge_title = result.get(
            "knowledge_title"
        )

        knowledge_similarity = result.get(
            "knowledge_similarity",
            0
        )

        solution = result.get(
            "solution",
            ""
        )

        ticket = result.get(
            "ticket"
        )

        escalation = result.get(
            "escalation"
        )


        # ====================================================
        # AI ANALYSIS
        # ====================================================

        st.header("🧠 AI Analysis")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Issue Category",
                category
            )

        with col2:

            st.metric(
                "AI Confidence",
                f"{confidence:.2f}%"
            )

        with col3:

            st.metric(
                "Priority",
                priority
            )


        # ====================================================
        # CONFIDENCE STATUS
        # ====================================================

        st.subheader("🎯 Confidence Analysis")

        if human_review:

            st.warning(
                "⚠️ Human review is recommended for this issue."
            )

        else:

            st.success(
                "✅ AI can handle this issue automatically."
            )


        st.divider()


        # ====================================================
        # RAG KNOWLEDGE
        # ====================================================

        st.header("📚 RAG Knowledge Retrieval")

        if knowledge_title:

            st.write(
                f"**Knowledge Article:** {knowledge_title}"
            )

            st.write(
                f"**Similarity Score:** "
                f"{knowledge_similarity:.3f}"
            )

            with st.expander(
                "📖 View Retrieved Knowledge"
            ):

                st.write(solution)

        else:

            st.warning(
                "⚠️ No relevant RAG knowledge result was found."
            )


        st.divider()


        # ====================================================
        # SOLUTION
        # ====================================================

        st.header("💡 Suggested Solution")

        if solution:

            st.info(solution)

        else:

            st.warning(
                "No solution was found in the knowledge base."
            )


        st.divider()


        # ====================================================
        # EXPLAINABLE AI
        # ====================================================

        st.header("🔍 Explainable AI")

        try:

            problem_vector = vectorizer.transform(
                [user_message]
            )

            predicted_class = model.predict(
                problem_vector
            )[0]

            feature_names = vectorizer.get_feature_names_out()

            coefficients = model.coef_

            class_names = model.classes_

            class_index = list(
                class_names
            ).index(predicted_class)

            feature_values = (
                problem_vector.toarray()[0]
            )

            class_coefficients = (
                coefficients[class_index]
            )

            contributions = (
                feature_values *
                class_coefficients
            )

            feature_contributions = []

            for index, value in enumerate(
                contributions
            ):

                if value > 0:

                    feature_contributions.append(
                        (
                            feature_names[index],
                            value
                        )
                    )

            feature_contributions.sort(
                key=lambda x: x[1],
                reverse=True
            )

            top_features = [
                item[0]
                for item in feature_contributions[:5]
            ]

            if top_features:

                st.write(
                    "The AI prediction was mainly influenced by:"
                )

                st.write(
                    " • ".join(top_features)
                )

            else:

                st.write(
                    "No strong contributing terms were identified."
                )

        except Exception as e:

            st.warning(
                f"Explainability information unavailable: {e}"
            )


        st.divider()


        # ====================================================
        # SUPPORT TICKET
        # ====================================================

        st.header("🎫 Support Ticket")

        if ticket:

            ticket_id = ticket.get(
                "Ticket ID",
                "Unknown"
            )

            ticket_status = ticket.get(
                "Status",
                "Open"
            )

            ticket_priority = ticket.get(
                "Priority",
                priority
            )

            ticket_category = ticket.get(
                "Category",
                category
            )

            st.success(
                f"✅ Ticket created successfully: **{ticket_id}**"
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.write(
                    f"**Ticket ID:** {ticket_id}"
                )

            with col2:

                st.write(
                    f"**Status:** {ticket_status}"
                )

            with col3:

                st.write(
                    f"**Priority:** {ticket_priority}"
                )

            st.write(
                f"**Category:** {ticket_category}"
            )

        else:

            st.warning(
                "No ticket was created."
            )


        st.divider()


        # ====================================================
        # HUMAN ESCALATION
        # ====================================================

        st.header("👨‍💻 Human Support Escalation")

        if escalation:

            st.error(
                "🚨 This issue has been escalated to human IT support."
            )

            if isinstance(
                escalation,
                dict
            ):

                message = escalation.get(
                    "message"
                )

                reason = escalation.get(
                    "reason"
                )

                if message:

                    st.write(
                        f"**Message:** {message}"
                    )

                if reason:

                    st.write(
                        f"**Reason:** {reason}"
                    )

        else:

            st.success(
                "✅ No human escalation is required."
            )


# ============================================================
# TICKET DASHBOARD
# ============================================================

st.divider()

st.header("📊 IT Helpdesk Dashboard")


TICKETS_FILE = "tickets.csv"


if os.path.exists(TICKETS_FILE):

    try:

        tickets = pd.read_csv(
            TICKETS_FILE
        )

        if not tickets.empty:

            # ----------------------------------------------
            # Dashboard Metrics
            # ----------------------------------------------

            total_tickets = len(tickets)

            open_tickets = len(
                tickets[
                    tickets["Status"] == "Open"
                ]
            )

            in_progress_tickets = len(
                tickets[
                    tickets["Status"] == "In Progress"
                ]
            )

            resolved_tickets = len(
                tickets[
                    tickets["Status"] == "Resolved"
                ]
            )


            # Calculate human review using
            # the same logic as the Agent

            human_review_count = 0

            for _, row in tickets.iterrows():

                try:

                    conf = float(
                        str(
                            row["Confidence"]
                        ).replace(
                            "%",
                            ""
                        )
                    )

                    row_category = row.get(
                        "Category",
                        ""
                    )

                    if (
                        row_category ==
                        "Security Issue"
                        or conf < 60
                    ):

                        human_review_count += 1

                except Exception:

                    pass


            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:

                st.metric(
                    "Total Tickets",
                    total_tickets
                )

            with col2:

                st.metric(
                    "Open",
                    open_tickets
                )

            with col3:

                st.metric(
                    "In Progress",
                    in_progress_tickets
                )

            with col4:

                st.metric(
                    "Resolved",
                    resolved_tickets
                )

            with col5:

                st.metric(
                    "Human Review",
                    human_review_count
                )


            st.divider()


            # ----------------------------------------------
            # Tickets by Category
            # ----------------------------------------------

            st.subheader(
                "📈 Tickets by Category"
            )

            category_counts = (
                tickets["Category"]
                .value_counts()
            )

            st.bar_chart(
                category_counts
            )


            st.divider()


            # ----------------------------------------------
            # Ticket Table
            # ----------------------------------------------

            st.subheader(
                "🎫 All Support Tickets"
            )

            st.dataframe(
                tickets,
                use_container_width=True
            )


            st.divider()


            # ----------------------------------------------
            # Update Ticket Status
            # ----------------------------------------------

            st.subheader(
                "🔄 Update Ticket Status"
            )

            if "Ticket ID" in tickets.columns:

                ticket_ids = tickets[
                    "Ticket ID"
                ].astype(str).tolist()

                selected_ticket = st.selectbox(
                    "Select Ticket",
                    ticket_ids
                )

                new_status = st.selectbox(
                    "Select New Status",
                    [
                        "Open",
                        "In Progress",
                        "Resolved"
                    ]
                )

                if st.button(
                    "Update Ticket Status"
                ):

                    update_result = (
                        update_ticket_status(
                            selected_ticket,
                            new_status
                        )
                    )

                    if update_result.get(
                        "success"
                    ):

                        st.success(
                            f"✅ Ticket "
                            f"{selected_ticket} "
                            f"updated to "
                            f"**{new_status}**."
                        )

                        st.rerun()

                    else:

                        st.error(
                            update_result.get(
                                "message",
                                "Unable to update ticket."
                            )
                        )

        else:

            st.info(
                "No support tickets available yet."
            )

    except Exception as e:

        st.error(
            f"Unable to read tickets.csv: {e}"
        )

else:

    st.info(
        "No support tickets have been created yet."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI IT Helpdesk Agent | "
    "Machine Learning + AI Agent + RAG + Tools + Explainable AI"
)