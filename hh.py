import streamlit as st
from PyPDF2 import PdfReader
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import json
import os
import pandas as pd
import re

# === Load lightweight LLM ===
@st.cache_resource
def load_llm():
    model_id = "google/flan-t5-base"  # CPU-friendly model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=1024)
    return HuggingFacePipeline(pipeline=pipe)

# === Extract text from PDF ===
def extract_text(pdf):
    reader = PdfReader(pdf)
    return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())

# === Prompt chain for resume extraction ===
def get_chain(llm):
    prompt = PromptTemplate(
        input_variables=["resume"],
        template="""
You are a resume parsing expert. Your task is to extract all available information from the resume text below into valid JSON.

Only output the JSON. Do not include any text before or after.

Use this structure:

{{
  "Name": "",
  "Roll Number": "",
  "Email": "",
  "Phone": "",
  "Education": [
    {{
      "Degree": "",
      "Institution": "",
      "Year": "",
      "CGPA_or_Percentage": ""
    }}
  ],
  "Skills": [],
  "Programming Languages": [],
  "Tools": [],
  "Libraries": [],
  "DevOps": [],
  "Certifications": [],
  "Projects": [
    {{
      "Title": "",
      "Role": "",
      "Duration": "",
      "Description": ""
    }}
  ],
  "Experience": [
    {{
      "Title": "",
      "Organization": "",
      "Duration": "",
      "Description": ""
    }}
  ],
  "Extra Curricular": []
}}

Resume Text:
{resume}
"""
    )
    return LLMChain(llm=llm, prompt=prompt)




# === JSON parse helper ===
import re

def parse_json_output(raw):
    try:
        # Remove backticks and markdown fences
        raw = raw.strip().replace("```json", "").replace("```", "")

        # Extract JSON using regex
        match = re.search(r"{.*}", raw, re.DOTALL)
        if match:
            json_str = match.group(0)
            return json.loads(json_str)

        st.error("⚠️ No valid JSON object found.")
        return None
    except json.JSONDecodeError as e:
        st.error(f"⚠️ Could not parse output as valid JSON: {e}")
        return None



# === Storage ===
def load_stored_data():
    if os.path.exists("parsed_resumes.json"):
        with open("parsed_resumes.json", "r") as f:
            return json.load(f)
    return []

def save_resume(parsed_data):
    all_data = load_stored_data()
    all_data.append(parsed_data)
    with open("parsed_resumes.json", "w") as f:
        json.dump(all_data, f, indent=2)

# === Streamlit App ===
def main():
    st.set_page_config("Resume Parser (CPU)", page_icon="🧠")
    st.title("🧠 Resume Parser (LLM-powered, CPU-friendly)")

    llm = load_llm()
    chain = get_chain(llm)

    uploaded_files = st.file_uploader("Upload one or more resumes (PDF)", type=["pdf"], accept_multiple_files=True)

    if uploaded_files and st.button("Parse Resumes"):
        for pdf in uploaded_files:
            with st.spinner(f"Parsing {pdf.name}..."):
                text = extract_text(pdf)
                if not text:
                    st.warning(f"No text found in {pdf.name}")
                    continue

                output = chain.run(resume=text)
                parsed = parse_json_output(output)

                if parsed:
                    parsed["Filename"] = pdf.name
                    save_resume(parsed)
                    st.success(f"✅ Parsed: {parsed.get('Name', 'Unknown')}")
                    st.json(parsed)
                else:
                    st.error(f"❌ Failed to parse structured JSON.")
                    st.text(output)

    # === View stored data ===
    st.subheader("📂 Parsed Resumes")
    resume_data = load_stored_data()

    if resume_data:
        for r in resume_data:
            with st.expander(f"👤 {r.get('Name', 'Unnamed')} ({r.get('Filename', '')})"):
                st.markdown(f"**Roll Number:** {r.get('Roll Number', '')}")
                st.markdown(f"**Email:** {r.get('Email', '')} | **Phone:** {r.get('Phone', '')}")
                st.markdown(f"**Skills:** {', '.join(r.get('Skills', []))}")
                st.markdown(f"**Programming Languages:** {', '.join(r.get('Programming Languages', []))}")
                st.markdown(f"**Tools:** {', '.join(r.get('Tools', []))}")
                st.markdown(f"**Libraries:** {', '.join(r.get('Libraries', []))}")
                st.markdown(f"**DevOps:** {', '.join(r.get('DevOps', []))}")
                st.markdown("**Certifications:**")
                for cert in r.get("Certifications", []):
                    st.markdown(f"- {cert}")
                st.markdown("**Education:**")
                for edu in r.get("Education", []):
                    st.markdown(f"- {edu.get('Degree')} from {edu.get('Institution')} ({edu.get('Year')}) — {edu.get('CGPA_or_Percentage')}")
                st.markdown("**Experience:**")
                for exp in r.get("Experience", []):
                    st.markdown(f"- {exp.get('Title')} at {exp.get('Organization')} ({exp.get('Duration')}): {exp.get('Description')}")
                st.markdown("**Projects:**")
                for proj in r.get("Projects", []):
                    st.markdown(f"- {proj.get('Title')} ({proj.get('Duration')}): {proj.get('Description')}")
                st.markdown("**Extra Curricular:**")
                for act in r.get("Extra Curricular", []):
                    st.markdown(f"- {act}")

        st.download_button(
            "📥 Download All as JSON",
            data=json.dumps(resume_data, indent=2),
            file_name="all_parsed_resumes.json",
            mime="application/json"
        )
    else:
        st.info("No resumes parsed yet.")

if __name__ == "__main__":
    main()
