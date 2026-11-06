import os 
from langchain_google_genai import ChatGoogleGenerativeAI
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
base_llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=api_key,
    thinking_level="high",
)

llm = RetryLLM(llm)