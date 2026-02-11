# src/demo_app.py
"""
Streamlit Demo App for Chief Complaint Classification
----------------------------------------------------
Input: free-text chief complaint
Output: multi-label and primary classification
"""

import streamlit as st
from preprocessing import clean_text
from labeling import classify_row

st.title("Chief Complaint Classification Demo")

# User input
complaint = st.text_area("Enter Chief Complaint:", height=100, placeholder="e.g., no fever but abdominal pain since 2 days")

if st.button("Classify"):
    if complaint:
        with st.spinner("Processing..."):
            # Step 1: Preprocess
            processed = clean_text(complaint)
            st.subheader("Processed Text")
            st.write(processed)

            # Step 2: Classify
            multi_labels, primary = classify_row(processed)

            # Display results
            st.subheader("Multi-Label Classification")
            st.write(", ".join(multi_labels) if multi_labels else "None")

            st.subheader("Primary Classification")
            st.write(primary)
    else:
        st.error("Please enter a complaint.")