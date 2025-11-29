import streamlit as st
import pandas as pd
import io

st.title("📁 File Uploader to DataFrame")
st.write("Upload your data files and preview them as DataFrames!")

# File uploader
uploaded_file = st.file_uploader(
    "Choose a file",
    type=["csv", "xlsx", "xls", "json", "parquet"],
    help="Supported formats: CSV, Excel (.xlsx, .xls), JSON, Parquet"
)

if uploaded_file is not None:
    try:
        # Get file extension
        file_extension = uploaded_file.name.split('.')[-1].lower()

        # Load file based on extension
        if file_extension == 'csv':
            df = pd.read_csv(uploaded_file)
        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(uploaded_file)
        elif file_extension == 'json':
            df = pd.read_json(uploaded_file)
        elif file_extension == 'parquet':
            df = pd.read_parquet(uploaded_file)
        else:
            st.error(f"Unsupported file type: {file_extension}")
            df = None

        if df is not None:
            # Display file info
            st.success(f"✅ Successfully loaded {uploaded_file.name}")

            # Show dataset information
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", df.shape[0])
            with col2:
                st.metric("Columns", df.shape[1])
            with col3:
                st.metric("Size", f"{uploaded_file.size / 1024:.2f} KB")

            # Display dataframe
            st.subheader("📊 Data Preview")
            st.dataframe(df, use_container_width=True)

            # Show column information
            with st.expander("ℹ️ Column Information"):
                col_info = pd.DataFrame({
                    'Column': df.columns,
                    'Type': df.dtypes.values,
                    'Non-Null Count': df.count().values,
                    'Null Count': df.isnull().sum().values
                })
                st.dataframe(col_info, use_container_width=True)

            # Show basic statistics
            with st.expander("📈 Statistics"):
                st.dataframe(df.describe(), use_container_width=True)

            # Download options
            st.subheader("💾 Download Options")
            col1, col2 = st.columns(2)

            with col1:
                # Download as CSV
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download as CSV",
                    data=csv,
                    file_name=f"{uploaded_file.name.split('.')[0]}_processed.csv",
                    mime="text/csv"
                )

            with col2:
                # Download as Excel
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False, sheet_name='Sheet1')
                excel_data = buffer.getvalue()

                st.download_button(
                    label="Download as Excel",
                    data=excel_data,
                    file_name=f"{uploaded_file.name.split('.')[0]}_processed.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        st.info("Please make sure the file is properly formatted and not corrupted.")
else:
    st.info("👆 Please upload a file to get started")

    # Show example
    with st.expander("💡 Example Usage"):
        st.markdown("""
        **Supported File Formats:**
        - **CSV** (.csv) - Comma-separated values
        - **Excel** (.xlsx, .xls) - Microsoft Excel files
        - **JSON** (.json) - JavaScript Object Notation
        - **Parquet** (.parquet) - Apache Parquet format

        **Features:**
        - 📊 Interactive data preview
        - 📈 Basic statistics
        - ℹ️ Column information and data types
        - 💾 Download processed data in CSV or Excel format
        """)
