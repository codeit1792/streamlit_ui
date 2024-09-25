import streamlit as st
from huggingface_hub import login
from vllm import LLM, SamplingParams

# Automatically log in to Hugging Face using the provided token
login(token='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')

# Streamlit UI Layout
st.set_page_config(page_title="QPiAI Chat", layout="wide")
st.title("🤖 QPiAI Chat")

# Sidebar for Model Configuration
st.sidebar.header("Model Configuration")

# Allow the user to input the model they want to use
model_name = st.sidebar.text_input(
    "Hugging Face Model", value="pavan01729/Merged_axolotl_llama3.1_GPT_Ai_5.0"
)

# Initialize the model with user-defined GPU memory utilization
gpu_memory = st.sidebar.slider("GPU Memory Utilization", min_value=0.1, max_value=1.0, value=0.5)

# Define a function to load the model
@st.cache_resource(show_spinner=False)
def load_model(model_name, gpu_memory):
    return LLM(model=model_name, gpu_memory_utilization=gpu_memory)

# Load the model when the app starts
llm = load_model(model_name, gpu_memory)
st.sidebar.success(f"Model {model_name} loaded with {gpu_memory * 100}% GPU memory utilization.")

# Conversation history stored in session state
if "conversation" not in st.session_state:
    st.session_state.conversation = []

# Clear conversation button
if st.sidebar.button("Clear Chat"):
    st.session_state.conversation = []

# Sampling parameters for generation
with st.sidebar.expander("Generation Settings"):
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.8)
    top_p = st.slider("Top-p (nucleus sampling)", min_value=0.1, max_value=1.0, value=0.95)
    max_tokens = st.slider("Max Tokens", min_value=10, max_value=500, value=100)
    repetition_penalty = st.slider("Repetition Penalty", min_value=0.5, max_value=2.0, value=1.2)

# Function to display chat bubbles with distinct styling
def display_conversation():
    for i, chat in enumerate(st.session_state.conversation):
        if chat["role"] == "user":
            st.markdown(
                f'<div style="background-color:#E2F0D9;padding:10px;border-radius:10px;margin-bottom:10px;text-align:right;">'
                f'<strong>You:</strong> {chat["text"]}</div>', unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div style="background-color:#F0F0F5;padding:10px;border-radius:10px;margin-bottom:10px;">'
                f'<strong>Model:</strong> {chat["text"]}</div>', unsafe_allow_html=True
            )

# Function to handle message input submission via Enter
def submit_message():
    user_input = st.session_state.user_input
    if user_input:
        # Append user input to conversation
        st.session_state.conversation.append({"role": "user", "text": user_input})

        with st.spinner("Model is thinking..."):
            # Define sampling parameters
            sampling_params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                repetition_penalty=repetition_penalty,
            )

            # Generate the text
            outputs = llm.generate([user_input], sampling_params)  # Pass the user_input as a list
            for output in outputs:
                generated_text = output.outputs[0].text

            # Append model response to conversation
            st.session_state.conversation.append({"role": "model", "text": generated_text})

        # Clear the input box after submission
        st.session_state.user_input = ""

# Chat UI Header
st.write("### Chat with your finetuned model")

# Container for conversation history
conversation_container = st.container()

# Display the conversation
with conversation_container:
    display_conversation()

# Static input area at the bottom of the chat, use Enter to submit
st.text_input(
    "You:", key="user_input", placeholder="Type your message here and press Enter to send...",
    on_change=submit_message,  # Call submit when Enter is pressed
)

# Auto-scroll to the bottom after each new message
st.write("")

