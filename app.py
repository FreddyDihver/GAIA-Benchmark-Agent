import os
import gradio as gr
from langchain_google_genai import ChatGoogleGenerativeAI
import spaces

from tools import *
from agent import *
from submitting import *
from model import *


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
        Once clicking on the "submit button, it can take quite some time ( this is the time for the agent to go through all the questions).
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
        print(f"   Runtime URL should be: https://{space_host_startup}")
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