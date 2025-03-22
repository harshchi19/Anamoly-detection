import streamlit as st
import fitz  # PyMuPDF for extracting text from PDFs
import google.generativeai as genai
import pandas as pd
import io  # For handling CSV download

# Set up Gemini API key
GEMINI_API_KEY = "AIzaSyAwY29cyESToWBGM3Rg2mEghTJUGyMaoJw"
genai.configure(api_key=GEMINI_API_KEY)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = "\n".join([page.get_text("text") for page in doc])
    return text

# Function to analyze anomalies using Gemini API
def analyze_anomalies_with_citation(text):
    prompt = f"""
    The following is a financial report extracted from a PDF. Identify anomalies such as:
    - Sudden revenue fluctuations
    - High unexplained expenses
    - Suspicious transactions
    - Unusual liabilities

    For each anomaly, provide:
    1. **Issue (Type of anomaly)**
    2. **Observations (Exact value from the report)**
    3. **Investee’s Comment (Possible explanation)**
    4. **Proposed Action (What should be done?)**
    
    Financial Report:
    {text}

    Respond with a structured textual summary followed by a table format:
    
    ### **Anomaly Detection Summary**
    - Key financial issues identified
    - Explanation of findings
    - Potential risks

    **Table Format**:
    | Issue | Observations | Investee’s Comment | Proposed Action |
    |-------|-------------|--------------------|-----------------|
    """
    
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    
    return response.text if response else "No anomalies detected."

# Function to parse Gemini API response into a structured DataFrame
def parse_anomalies_to_table(response_text):
    rows = []
    lines = response_text.split("\n")
    
    for line in lines:
        if "|" in line and "Issue" not in line:  # Ignore header row
            columns = [col.strip() for col in line.split("|")[1:-1]]  # Ignore first and last empty columns
            if len(columns) == 4:  # Ensure valid row
                rows.append(columns)
    
    return pd.DataFrame(rows, columns=["Issue", "Observations", "Investee’s Comment", "Proposed Action"])

# Function to extract textual summary from the response
def extract_summary(response_text):
    return response_text.split("**Table Format**:")[0].strip() if "**Table Format**:" in response_text else response_text

# Streamlit UI
st.set_page_config(page_title="Financial Anomaly Detection", layout="wide")

st.markdown(
    "<h1 style='text-align: center; color: #FF5733;'>📊 AI-Powered Financial Anomaly Detection</h1>", 
    unsafe_allow_html=True
)
st.markdown("### Upload a **financial report PDF** to analyze potential anomalies and risks.")

# File uploader
uploaded_file = st.file_uploader("📂 Upload PDF", type=["pdf"])

if uploaded_file:
    st.success("✅ File uploaded successfully!")
    
    # Extract text
    with st.spinner("🔍 Extracting text from PDF..."):
        extracted_text = extract_text_from_pdf(uploaded_file)

    # Display extracted text
    with st.expander("📄 **Extracted Text from PDF**", expanded=False):
        st.text_area("", extracted_text, height=250)

    # Analyze anomalies button
    if st.button("🚀 Analyze Anomalies"):
        with st.spinner("🤖 Analyzing anomalies using Gemini API..."):
            anomalies_report = analyze_anomalies_with_citation(extracted_text)

        # Extract summary and table data
        summary_text = extract_summary(anomalies_report)
        anomalies_df = parse_anomalies_to_table(anomalies_report)

        # Display anomaly detection summary
        st.subheader("📜 **Anomaly Detection Summary**")
        st.markdown(f"<div style='background-color: #F7F7F7; padding: 15px; border-radius: 10px;'>{summary_text}</div>", unsafe_allow_html=True)

        # Display structured table
        st.subheader("📌 **Anomaly Detection Report with Citation**")
        
        if not anomalies_df.empty:
            st.dataframe(anomalies_df, use_container_width=True)

            # Create CSV buffer
            csv_buffer = io.StringIO()
            anomalies_df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            
            # Download button
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name="anomaly_detection_report.csv",
                mime="text/csv"
            )
        else:
            st.warning("⚠️ No anomalies detected.")
