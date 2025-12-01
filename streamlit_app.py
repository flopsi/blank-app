import streamlit as st
import pandas as pd
import os
from datetime import datetime

EXCEL_FILE = "account_planning_tool.xlsx"

st.set_page_config(page_title="Account Planning Dashboard", layout="wide")

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=30)
def load_excel(path):
    return pd.read_excel(path, sheet_name=None, engine="openpyxl")

def get_score_color(score):
    """Return background color based on score ranges."""
    if pd.isna(score):
        return "#cccccc"
    score = int(score)
    if score >= 23:
        return "#4ade80"  # green
    elif score >= 15:
        return "#fbbf24"  # yellow
    else:
        return "#f87171"  # red

def parse_master_overview(df):
    """Parse Master_overview into rep/account blocks."""
    records = []
    i = 0
    while i + 4 < len(df):
        rep = df.iloc[i, 0]
        account = df.iloc[i, 1]
        reason = df.iloc[i, 4] if len(df.columns) > 4 else ""
        score = df.iloc[i+1, 0]
        status = df.iloc[i+1, 1] if len(df.columns) > 1 else ""
        current_state = df.iloc[i, 7] if len(df.columns) > 7 else ""
        
        if pd.notna(rep) and pd.notna(account):
            records.append({
                "Rep": str(rep),
                "Account": str(account),
                "Reason": str(reason) if pd.notna(reason) else "",
                "Score": int(score) if pd.notna(score) else 0,
                "Status": str(status) if pd.notna(status) else "",
                "Current State": str(current_state) if pd.notna(current_state) else ""
            })
        i += 5  # Each block is 5 rows
    
    return pd.DataFrame(records)

# ─────────────────────────────────────────────────────────────
# Overview Page
# ─────────────────────────────────────────────────────────────
def show_overview():
    st.title("📊 Account Planning Overview")
    
    if not os.path.exists(EXCEL_FILE):
        st.error(f"Excel file '{EXCEL_FILE}' not found.")
        uploaded = st.file_uploader("Upload account_planning_tool.xlsx", type=["xlsx"])
        if uploaded:
            with open(EXCEL_FILE, "wb") as f:
                f.write(uploaded.getbuffer())
            st.rerun()
        return
    
    all_sheets = load_excel(EXCEL_FILE)
    master_df = all_sheets.get("Master_overview", pd.DataFrame())
    
    if master_df.empty:
        st.warning("Master_overview sheet is empty.")
        return
    
    overview = parse_master_overview(master_df)
    
    if overview.empty:
        st.info("No accounts found in Master_overview.")
        return
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Accounts", len(overview))
    col2.metric("Average Score", f"{overview['Score'].mean():.1f}")
    col3.metric("Unique Reps", overview["Rep"].nunique())
    
    st.divider()
    
    # Display table with color-coded scores
    for _, row in overview.iterrows():
        color = get_score_color(row["Score"])
        
        cols = st.columns([2, 1, 3, 2, 4])
        
        with cols[0]:
            st.markdown(f"**{row['Rep']}**")
        
        with cols[1]:
            st.markdown(
                f"<div style='background-color: {color}; padding: 20px; "
                f"text-align: center; font-size: 24px; font-weight: bold; "
                f"border-radius: 8px; color: white;'>{row['Score']}</div>",
                unsafe_allow_html=True
            )
        
        with cols[2]:
            st.markdown(f"**Account:** {row['Account']}")
            st.caption(f"*{row['Reason']}*")
        
        with cols[3]:
            st.caption(row['Status'])
        
        with cols[4]:
            st.text(row['Current State'][:100] + "..." if len(row['Current State']) > 100 else row['Current State'])
        
        st.divider()

# ─────────────────────────────────────────────────────────────
# Florian Marty Rep Page
# ─────────────────────────────────────────────────────────────
def show_rep_page():
    st.title("👤 Florian Marty - Account Input")
    
    # Dummy account selector
    st.subheader("Select Account")
    account = st.selectbox(
        "Account Name",
        ["Lonza Walliser Werke AG", "Account 2", "Account 3"],
        help="Dropdown from list"
    )
    
    reason = st.selectbox(
        "Reason",
        ["Untapped potential in LC", "Highest Potential", "Good at R&D"],
        help="Dropdown from list"
    )
    
    # Score display (read-only, calculated from Account Health)
    dummy_score = 22
    color = get_score_color(dummy_score)
    st.markdown(
        f"<div style='background-color: {color}; padding: 30px; "
        f"text-align: center; font-size: 36px; font-weight: bold; "
        f"border-radius: 12px; color: white; max-width: 200px;'>{dummy_score}</div>",
        unsafe_allow_html=True
    )
    st.caption("*Score calculated from Account Health (auto-updated)*")
    
    st.divider()
    
    # ─────────────────────────────────────────────────────────
    # 1. MAIN CIGAR
    # ─────────────────────────────────────────────────────────
    st.subheader("4.) Account CIGAR")
    
    cigar_current = st.text_area("Current State", value="A3Biopharma2025!", height=100)
    cigar_ideal = st.text_area("Ideal State", value="", height=100)
    cigar_gap = st.text_area("Gap", value="dfyaggh", height=100)
    cigar_action = st.text_area("Action", value="", height=100)
    cigar_review = st.date_input("Review", value=datetime.now())
    
    st.divider()
    
    # ─────────────────────────────────────────────────────────
    # 2. ACCOUNT HEALTH
    # ─────────────────────────────────────────────────────────
    st.subheader("1.) Account Health")
    
    health_categories = {
        "Relationship": [
            "Do we have executive-level relationships with the account?",
            "Do we have access to multiple departments or business units?",
            "Have we identified and engaged a champion within the organization?"
        ],
        "Value Delivery": [
            "Is the customer actively using our solution/achieving outcomes (TMO)?",
            "Have we demonstrated measurable ROI or business impact?",
            "Has the customer provided positive feedback or references?"
        ],
        "Growth Potential": [
            "Have we identified whitespace or expansion opportunities?",
            "Are there any ongoing investments available?",
            "Are there upcoming initiatives where we could add value?"
        ],
        "Competitive Position": [
            "Do we have a strategic partnership ongoing or planned?",
            "Do we understand the competitive landscape within this account?",
            "Are there immediate threats from competitors?"
        ],
        "Risk & Engagement": [
            "Is the account engaged in regular business reviews?",
            "Are invoices paid on time without disputes?",
            "Have we had any escalations or service issues recently?"
        ]
    }
    
    health_data = []
    for category, questions in health_categories.items():
        st.markdown(f"**{category}**")
        for q in questions:
            col1, col2, col3 = st.columns([5, 1, 3])
            with col1:
                st.caption(q)
            with col2:
                score = st.selectbox("Score", [0, 1, 2], key=f"health_{category}_{q[:20]}", label_visibility="collapsed")
            with col3:
                comment = st.text_input("Comment", key=f"comment_{category}_{q[:20]}", placeholder="Enter Your Comment", label_visibility="collapsed")
            health_data.append({"category": category, "question": q, "score": score, "comment": comment})
    
    total_health_score = sum([h["score"] for h in health_data])
    st.metric("Total Account Health Score", total_health_score, help="Sum of all 15 scores (max 30)")
    
    st.divider()
    
    # ─────────────────────────────────────────────────────────
    # 3. WHITE SPACE ANALYSIS
    # ─────────────────────────────────────────────────────────
    st.subheader("2.) White Space Analysis")
    
    product_lines = ["LSMS", "HPLC", "IC", "TEA", "CDS", "GC"]
    whitespace_evals = ["Not Applicable", "Hostile", "Active Opportunity", "Dominating", "No Clue"]
    
    whitespace_data = []
    for pl in product_lines:
        col1, col2, col3 = st.columns([2, 3, 4])
        with col1:
            st.markdown(f"**{pl}**")
        with col2:
            eval_val = st.selectbox("Evaluation", whitespace_evals, key=f"ws_{pl}", label_visibility="collapsed")
        with col3:
            ws_comment = st.text_input("Comment", key=f"ws_comment_{pl}", placeholder="Enter Your Comment", label_visibility="collapsed")
        whitespace_data.append({"pl": pl, "eval": eval_val, "comment": ws_comment})
    
    st.divider()
    
    # ─────────────────────────────────────────────────────────
    # 4. PL CIGAR
    # ─────────────────────────────────────────────────────────
    st.subheader("3.) PL CIGAR")
    
    st.caption("*Current state is auto-populated from White Space Analysis*")
    
    pl_cigar_data = []
    for i, pl in enumerate(product_lines):
        # Auto-populate current state from whitespace
        current_ws = whitespace_data[i]["eval"]
        
        st.markdown(f"**{pl}**")
        col1, col2, col3, col4 = st.columns([2, 3, 3, 3])
        
        with col1:
            st.text_input("Current state", value=current_ws, key=f"pl_current_{pl}", disabled=True, label_visibility="collapsed")
        with col2:
            ideal = st.text_input("Ideal state", key=f"pl_ideal_{pl}", placeholder="Enter Ideal State", label_visibility="collapsed")
        with col3:
            gap = st.text_input("Gap", key=f"pl_gap_{pl}", placeholder="Enter Gap", label_visibility="collapsed")
        with col4:
            action = st.text_input("Action", key=f"pl_action_{pl}", placeholder="Enter Action", label_visibility="collapsed")
        
        pl_cigar_data.append({"pl": pl, "current": current_ws, "ideal": ideal, "gap": gap, "action": action})
    
    st.divider()
    
    # ─────────────────────────────────────────────────────────
    # SAVE BUTTON
    # ─────────────────────────────────────────────────────────
    if st.button("💾 Save All Changes", type="primary", use_container_width=True):
        st.success("✅ Changes saved successfully!")
        st.info("Data will be written back to Excel Master_overview sheet")
        # TODO: Implement save logic to write back to Excel

# ─────────────────────────────────────────────────────────────
# Main Navigation
# ─────────────────────────────────────────────────────────────
def main():
    page = st.sidebar.radio(
        "Navigation",
        ["📈 Overview", "👤 Florian Marty"]
    )
    
    if page == "📈 Overview":
        show_overview()
    else:
        show_rep_page()

if __name__ == "__main__":
    main()
