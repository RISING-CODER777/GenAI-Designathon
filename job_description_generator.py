import boto3
import streamlit as st

#Bedrock model for text generation
from langchain.llms.bedrock import Bedrock
from langchain.prompts import PromptTemplate

# Initialize Bedrock Client
bedrock = boto3.client(service_name="bedrock-runtime")

# Function to initialize the LLM (e.g., Claude)
def get_claude_llm():
    llm = Bedrock(
        model_id="anthropic.claude-v2",
        client=bedrock,
        model_kwargs={"max_tokens_to_sample": 512}
    )
    return llm

# Prompt template
prompt_template = """
Human: You are an HR assistant. Generate a professional and detailed job description based on the following keywords:
Keywords: {keywords}

Include:
- Job Title
- Responsibilities
- Required Skills
- Qualifications

Assistant:
"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["keywords"])

# generate job descriptions
def generate_job_description(llm, keywords):
    input_text = PROMPT.format(keywords=keywords)
    response = llm(input_text)
    return response

#Streamlit app
def main():
    st.set_page_config(page_title="Job Description Generator", layout="wide")

    st.title("💼 Job Description Generator for HR Professionals")
    st.markdown("Generate tailored job descriptions by providing relevant keywords.")

    # Input field for keywords
    user_keywords = st.text_input("Enter keywords for the job description (e.g., Python, AI, team management)", "")

    if st.button("Generate Job Description"):
        if user_keywords:
            with st.spinner("Generating job description..."):
                llm = get_claude_llm()
                job_description = generate_job_description(llm, user_keywords)
                st.markdown("### Generated Job Description")
                st.write(job_description)
                st.success("Job description generated successfully!")
        else:
            st.error("Please enter some keywords to proceed.")

if __name__ == "__main__":
    main()
