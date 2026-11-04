import os
import gradio as gr
from langchain_google_genai import ChatGoogleGenerativeAI
import spaces

from tools import *
from agent import *
from submitting import *
from retry_llm import RetryLLM


#########
## API ##
#########


if "GEMINI_API_KEY" in os.environ:
    api_key = os.environ["GEMINI_API_KEY"]
else:
    api_key = input("Enter your GEMINI API key: ")



##########################
## LLM with Retry Logic ##
##########################
llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    google_api_key=api_key,
    thinking_level="high",
)

llm = RetryLLM(llm)


##################################
## Submission and ZeroGPU Check ##
##################################
def run_submission(profile: gr.OAuthProfile | None):
    return run_and_submit_all(profile, llm)

@spaces.GPU
def zerogpu_check():
    return "ZeroGPU handler registered"

############################
## Build Gradio Interface ##
############################
with gr.Blocks() as demo:
    gr.Markdown("# Basic Agent Evaluation Runner")
    gr.Markdown(
        """
        **Instructions:**

        1.  Please clone this space, then modify the code to define your agent's logic, the tools, the necessary packages, etc ...
        2.  Log in to your Hugging Face account using the button below. This uses your HF username for submission.
        3.  Click 'Run Evaluation & Submit All Answers' to fetch questions, run your agent, submit answers, and see the score.

        ---
        **Disclaimers:**
        Once clicking on the "submit button, it can take quite some time ( this is the time for the agent to go through all the questions).
        This space provides a basic setup and is intentionally sub-optimal to encourage you to develop your own, more robust solution. For instance for the delay process of the submit button, a solution could be to cache the answers and submit in a seperate action or even to answer the questions in async.
        """
    )

    # Zero GPU Handler
    gpu_check_button = gr.Button(visible=False)
    gpu_check_output = gr.Textbox(visible=False)
    
    gpu_check_button.click(
        fn=zerogpu_check,
        inputs=[],
        outputs=gpu_check_output
    )

    # Build
    gr.LoginButton()

    run_button = gr.Button("Run Evaluation & Submit All Answers")

    status_output = gr.Textbox(label="Run Status / Submission Result", lines=5, interactive=False)
    # Removed max_rows=10 from DataFrame constructor
    results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)

    run_button.click(
        fn=run_submission,
        outputs=[status_output, results_table]
    )

if __name__ == "__main__":
    print("\n" + "-"*30 + " App Starting " + "-"*30)
    # Check for SPACE_HOST and SPACE_ID at startup for information
    space_host_startup = os.getenv("SPACE_HOST")
    space_id_startup = os.getenv("SPACE_ID") # Get SPACE_ID at startup

    if space_host_startup:
        print(f"✅ SPACE_HOST found: {space_host_startup}")
        print(f"   Runtime URL should be: https://{space_host_startup}.hf.space")
    else:
        print("ℹ️  SPACE_HOST environment variable not found (running locally?).")

    if space_id_startup: # Print repo URLs if SPACE_ID is found
        print(f"✅ SPACE_ID found: {space_id_startup}")
        print(f"   Repo URL: https://huggingface.co/spaces/{space_id_startup}")
        print(f"   Repo Tree URL: https://huggingface.co/spaces/{space_id_startup}/tree/main")
    else:
        print("ℹ️  SPACE_ID environment variable not found (running locally?). Repo URL cannot be determined.")

    print("-"*(60 + len(" App Starting ")) + "\n")

    print("Launching Gradio Interface for Basic Agent Evaluation...")
    demo.launch(debug=True, share=False)