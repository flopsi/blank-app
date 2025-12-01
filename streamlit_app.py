import streamlit as st
import pandas as pd
from io import BytesIO
import os

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
EXCEL_FILE = "account_planning_tool.xlsx"

st.set_page_config(page_title="Account Planning Dashboard", layout="wide")

# ─────────────────────────────────────────────────────────────────────────────
# LOAD & CACHE EXCEL
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def load_excel(path):
    """Load all sheets into a dict of DataFrames."""
    return pd.read_excel(path, sheet_name=None, engine="openpyxl")

def save_excel(data_dict, path):
    """Write all sheets back to Excel."""
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, df in data_dict.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

# ─────────────────────────────────────────────────────────────────────────────
# PARSE MASTER OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
def parse_master_overview(df):
    """Extract rep/account/reason/status from Master_overview."""
    records = []
    for i in range(0, len(df), 7):  # Each rep block is ~7 rows
        if i + 4 >= len(df):
            break
        rep_name = df.iloc[i, 0]
        account_name = df.iloc[i, 1]
        reason = df.iloc[i, 4]
        status_score = df.iloc[i+1, 0]
        current_state = df.iloc[i, 7] if len(df.columns) > 7 else ""
        if pd.notna(rep_name) and rep_name.strip():
            records.append({
                "Rep": rep_name,
                "Account": account_name,
                "Reason": reason,
                "Status": status_score,
                "Current State": current_state
            })
    return pd.DataFrame(records)

# ─────────────────────────────────────────────────────────────────────────────
# PARSE REP SHEET (Account Health Scores)
# ─────────────────────────────────────────────────────────────────────────────
def parse_rep_sheet(df):
    """Extract account name, health scores, white space, and CIGAR from rep sheet."""
    data = {}
    # Account name typically at row 0, col 2
    account_row = df[df.iloc[:, 1] == "Account Name:"]
    if not account_row.empty:
        data["Account Name"] = account_row.iloc[0, 2]
    
    # Find Account Health section
    health_start = df[df.iloc[:, 1] == "Account Health"].index
    if not health_start.empty:
        start = health_start[0]
        scores = []
        for i in range(start+2, min(start+20, len(df))):
            score_val = df.iloc[i, 4]
            if pd.notna(score_val) and str(score_val).isdigit():
                scores.append(int(score_val))
        data["Health Score"] = sum(scores) if scores else 0
    
    # White Space Analysis
    ws_start = df[df.iloc[:, 1] == "White Space Analysis"].index
    if not ws_start.empty:
        start = ws_start[0]
        ws_data = []
        for i in range(start+2, min(start+10, len(df))):
            product = df.iloc[i, 2]
            evaluation = df.iloc[i, 3]
            if pd.notna(product):
                ws_data.append({"Product Line": product, "Evaluation": evaluation})
        data["White Space"] = pd.DataFrame(ws_data) if ws_data else pd.DataFrame()
    
    # Account CIGAR
    cigar_start = df[df.iloc[:, 1] == "Account CIGAR"].index
    if not cigar_start.empty:
        start = cigar_start[0]
        cigar = {}
        for i in range(start, min(start+6, len(df))):
            key = df.iloc[i, 2]
            val = df.iloc[i, 3]
            if pd.notna(key):
                cigar[key] = val
        data["CIGAR"] = cigar
    
    return data

# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.title("📊 Account Planning Dashboard")
    
    # Load data
    if not os.path.exists(EXCEL_FILE):
        st.error(f"File {EXCEL_FILE} not found. Upload it to continue.")
        uploaded = st.file_uploader("Upload Excel file", type=["xlsx"])
        if uploaded:
            with open(EXCEL_FILE, "wb") as f:
                f.write(uploaded.getbuffer())
            st.rerun()
        return
    
    all_sheets = load_excel(EXCEL_FILE)
    
    # Sidebar navigation
    page = st.sidebar.radio("Navigate", ["📈 Overview", "👤 Rep View", "💬 AI Chat"])
    
    # ─────────────────────────────────────────────────────────────────────────
    # OVERVIEW PAGE
    # ─────────────────────────────────────────────────────────────────────────
    if page == "📈 Overview":
        st.header("Master Overview")
        master_df = all_sheets.get("Master_overview", pd.DataFrame())
        overview = parse_master_overview(master_df)
        if not overview.empty:
            st.dataframe(overview, use_container_width=True)
            
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Accounts", len(overview))
            col2.metric("Avg Status Score", f"{overview['Status'].mean():.1f}")
            col3.metric("Unique Reps", overview["Rep"].nunique())
        else:
            st.info("No data in Master_overview sheet.")
    
    # ─────────────────────────────────────────────────────────────────────────
    # REP VIEW PAGE
    # ─────────────────────────────────────────────────────────────────────────
    elif page == "👤 Rep View":
        rep_sheets = [s for s in all_sheets.keys() if s not in ["Master_overview", "helper_sheet"]]
        if not rep_sheets:
            st.warning("No rep sheets found.")
            return
        
        selected_rep = st.sidebar.selectbox("Select Rep Sheet", rep_sheets)
        st.header(f"Rep: {selected_rep}")
        
        rep_df = all_sheets[selected_rep]
        parsed = parse_rep_sheet(rep_df)
        
        # Display account info
        st.subheader(f"Account: {parsed.get('Account Name', 'N/A')}")
        st.metric("Account Health Score", parsed.get("Health Score", 0))
        
        # CIGAR
        if "CIGAR" in parsed:
            st.subheader("Account CIGAR")
            cigar = parsed["CIGAR"]
            for k, v in cigar.items():
                st.text_input(k, value=v if pd.notna(v) else "", key=f"cigar_{k}")
        
        # White Space
        if "White Space" in parsed and not parsed["White Space"].empty:
            st.subheader("White Space Analysis")
            edited_ws = st.data_editor(parsed["White Space"], use_container_width=True, num_rows="dynamic")
        
        # Save button (simplified - in production, map edits back to Excel structure)
        if st.button("💾 Save Changes"):
            st.success("Changes saved! (In production, write back to Excel)")
            # For full implementation: reconstruct rep_df from edited values and call save_excel
    
    # ─────────────────────────────────────────────────────────────────────────
    # AI CHAT PAGE
    # ─────────────────────────────────────────────────────────────────────────
    elif page == "💬 AI Chat":
        st.header("AI Account Assistant")
        
        # Select account context
        master_df = all_sheets.get("Master_overview", pd.DataFrame())
        overview = parse_master_overview(master_df)
        if overview.empty:
            st.warning("No accounts found.")
            return
        
        selected_account = st.selectbox("Select Account", overview["Account"].unique())
        account_data = overview[overview["Account"] == selected_account].iloc[0]
        
        st.info(f"**Account:** {selected_account}\n\n**Rep:** {account_data['Rep']}\n\n**Reason:** {account_data['Reason']}")
        
        # Chat interface
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        
        if prompt := st.chat_input("Ask about this account..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Build context for Perplexity
            context = f"Account: {selected_account}\nRep: {account_data['Rep']}\nReason: {account_data['Reason']}\nCurrent State: {account_data['Current State']}"
            
            # Placeholder response (integrate Perplexity API)
            response = f"[AI Response about {selected_account}]\n\nContext:\n{context}\n\nQuestion: {prompt}\n\n*To enable real AI responses, integrate Perplexity API with your API key.*"
            
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)

if __name__ == "__main__":
    main()
