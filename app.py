"""
Vireo Audio – Refund Analysis Agent (Streamlit UI) - 
"""

import streamlit as st
import plotly.io as pio
from langchain_core.messages import HumanMessage, AIMessage
from agent.agent import create_refund_agent

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Vireo Refund Agent",
    page_icon="📊",
    layout="wide"
)

st.title("Vireo Audio – Refund Analysis Agent")
st.caption("Ask questions about refunds by reason, by agent, quarters, and get visualizations.")

# ----------------------------------------------------------------------
# Initialize
# ----------------------------------------------------------------------
if "agent" not in st.session_state:
    try:
        with st.spinner("Loading agent..."):
            agent = create_refund_agent()
            if agent is None:
                st.error("create_refund_agent() returned None")
                st.stop()
            st.session_state.agent = agent
        st.session_state.messages = []
        st.session_state.langgraph_messages = []
    except Exception as e:
        st.error("Failed to create agent")
        st.exception(e)
        st.stop()

# ----------------------------------------------------------------------
# Helper: process a user message (used by both chat input and sidebar)
# ----------------------------------------------------------------------
def process_user_message(prompt: str):
    # Add user message to display
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.langgraph_messages.append(HumanMessage(content=prompt))

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing data..."):
            try:
                result = st.session_state.agent.invoke({
                    "messages": st.session_state.langgraph_messages
                })

                # Update memory
                st.session_state.langgraph_messages = result["messages"]

                # Look through ALL messages for a visualization
                answer = None
                fig = None

                for msg in reversed(result["messages"]):
                    content = getattr(msg, "content", None)
                    if not content:
                        continue

                    if isinstance(content, str) and content.startswith("VISUALIZATION_SUCCESS::"):
                        try:
                            parts = content.split("::", 2)
                            fig = pio.from_json(parts[2])
                            answer = f"Here's the visualization you requested:"
                            break
                        except Exception as e:
                            answer = f"Chart generated but could not be rendered: {e}"
                            break

                # If no visualization found, take the last AI message
                if answer is None:
                    last_msg = result["messages"][-1]
                    answer = last_msg.content

                # Display
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown(answer)
                else:
                    st.markdown(answer)

                # Save to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer if fig is None else "VISUALIZATION_SUCCESS::chart::" + parts[2]
                })

            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.exception(e)


# ----------------------------------------------------------------------
# Sidebar – example questions (now auto-sends)
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("Example Questions")
    examples = [
        "What is the total refund amount for 2025Q1?",
        "Show me the top 5 agents who gave the most refunds",
        "Compare 2025Q1 vs 2025Q2",
        "Give me refund breakdown by reason code for 2025Q4",
        "Show a bar chart of refunds by reason for 2025Q3",
        "Visualize top 10 agents by refund amount in 2026Q1",
        "Which agent issued the most refunds overall?",
        "Give me monthly refunds by agent for 2025-03"
    ]

    for example in examples:
        if st.button(example, use_container_width=True, key=f"ex_{example}"):
            process_user_message(example)
            st.rerun()

    st.divider()
    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.langgraph_messages = []
        st.rerun()


# ----------------------------------------------------------------------
# Display chat history
# ----------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        content = msg["content"]

        if isinstance(content, str) and content.startswith("VISUALIZATION_SUCCESS::"):
            try:
                parts = content.split("::", 2)
                fig = pio.from_json(parts[2])
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Could not render chart: {e}")
        else:
            st.markdown(content)


# ----------------------------------------------------------------------
# Chat input
# ----------------------------------------------------------------------
if prompt := st.chat_input("Ask about refunds..."):
    process_user_message(prompt)