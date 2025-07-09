import gradio as gr
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
import os

# --- Tool Initialization ---
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

search = DuckDuckGoSearchRun(name="Search")
tools = [search, arxiv, wiki]

# --- Chat Logic ---
def chat_function(message, history, api_key):
    api_key = api_key.strip()

    if api_key.lower() == "master":
        api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        gr.Warning("Please enter your Groq API Key.")
        return "", history

    try:
        llm = ChatGroq(
            groq_api_key=api_key,
            model_name="Llama3-8b-8192",
            streaming=True
        )

        agent = initialize_agent(
            tools,
            llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            handling_parsing_errors=True,
            verbose=True
        )

        response = agent.run(message)
        history.append((message, response))
        return "", history

    except Exception as e:
        gr.Error(f"An error occurred: {e}")
        return "", history

# --- Gradio Interface ---
with gr.Blocks(theme=gr.themes.Soft(), title="LangChain Search Bot") as demo:
    gr.Markdown("# 🔎 LangChain - Chat with Search")
    gr.Markdown(
        """
        This application uses a LangChain agent to answer questions by searching the web with DuckDuckGo, Wikipedia, and Arxiv.
        Enter your Groq API key, ask a question, and see the agent work.
        """
    )

    with gr.Row():
        api_key_input = gr.Textbox(
            label="Groq API Key",
            placeholder="Enter your Groq API key here...",
            type="password",
            lines=1
        )

    chatbot = gr.Chatbot(
        label="Chat History",
        bubble_full_width=False,
        avatar_images=(None, "download.png"),
        value=[[None, "Hi, I'm a chatbot who can search the web. How can I help you?"]]
    )

    message_input = gr.Textbox(
        label="Your Message",
        placeholder="e.g., What is machine learning?",
        lines=1
    )

    clear_button = gr.ClearButton([message_input, chatbot], value="Clear Chat")

    message_input.submit(
        chat_function,
        inputs=[message_input, chatbot, api_key_input],
        outputs=[message_input, chatbot]
    )

if __name__ == "__main__":
    demo.launch(debug=True)
