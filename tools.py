import os
import pandas as pd
from pathlib import Path
from langchain.tools import tool
from googleapiclient.discovery import build
from langchain_community.document_loaders import WebBaseLoader
from urllib.parse import urlparse, parse_qs

from ddgs import DDGS
from ddgs.exceptions import DDGSException
from langchain_core.messages import HumanMessage
import mimetypes

from model import *


###############################
## Web Search and Read Tools ##
###############################
@tool
def websearch_tool(query: str) -> str:
    """Search the web and return the results of the query.

    args:
        query (str): The search query for the web.
    returns:
        str: The search results or a message if no results are found.
    """
    try:
        results = DDGS(timeout=10).text(query, max_results=2)
        if not results:
            return "Search returned no results. Try a different query."

        return results
    except DDGSException as e:
        return f"Search failed or returned no results: {e}. Try a different query."


@tool
def visit_website_tool(url: str) -> str:
    """Retrieve webpage content or YouTube metadata, description and comments."""
    try:
        docs = WebBaseLoader(url).load()
        return "\n\n".join(doc.page_content for doc in docs) if docs else "No content found."

    except Exception as e:
        return f"Could not retrieve content from {url}: {e}"
    
@tool
def search_and_read_web_tool(query: str) -> str:
    """Search the web, then read the pages returned by the search.

    Use this as the default and usually only tool for web research
    when no direct URL is provided. The result includes page content,
    titles, and URLs. Do not call websearch_tool separately first.
    Do NOT use this tool to retrieve or investigate information that
    should be contained in a GAIA task attachment.
    
    args:
        query (str): The query to search for.
    returns:
        str: Content of the first webpages in the search
    """
    try:
        results = websearch_tool.invoke({"query": query})
    except Exception as e:
        print(f"Web search failed: {e}")
        return f"Web search failed: {e}"

    if not isinstance(results, list) or not results:
        print('No usable search results')
        return f"No usable search results. Search returned: {results}"

    pages = []
    url = results[0].get("href")
    if url:
        try:
            content = visit_website_tool.invoke({"url": url})
            pages.append(content)
            print('Searched the web')
        except Exception as e:
            pages.append(f"Could not read {url}: {e}")
            print(f"Could not read {url}: {e}")

    return "\n\n".join(pages) if pages else "No result pages could be read."


##########################
## Gaia Attachment Tool ##
##########################
from huggingface_hub import hf_hub_download
@tool
def download_gaia_attachment(file_name: str) -> str:
    """
    Download an attachment from the GAIA validation dataset.
    IMPORTANT: This tool MUST be called before any web search whenever the question mentions an attached file or filename.

    Args:
        file_name: Exact attachment filename provided by the GAIA task.

    Returns:
        Local path to the downloaded attachment.
    """

    try:
        if Path(file_name).name != file_name:
            return "Invalid filename."

        print(f"Downloading attachment: {file_name}")       
        print(f"Working directory: {os.getcwd()}")       
        print(f"HF token available: {bool(os.getenv('HF_TOKEN'))}")    
        file_path = hf_hub_download(repo_id="gaia-benchmark/GAIA",
                                    repo_type="dataset",
                                    filename=f"2023/validation/{file_name}",
                                    local_dir="downloads",
                                    token=os.getenv("HF_TOKEN"))
        print(f"Downloaded path: {file_path}")
        print(f"File exists: {Path(file_path).exists()}")
        return str(file_path)

    except Exception as e:
        print(f"Attachment download failed: {type(e).__name__}: {e}")
        return f"Error downloading attachment: {type(e).__name__}: {e}"

######################
## Read files tools ##
######################
@tool 
def read_text_tool(file_path: str) -> str:
    """Read a text file (python, script, text etc.) and return its content
    args:
        file_path (str): The path to the  file.
    returns:
            str: The content of the file or a message if no content is found.
    """
    try:
        path = Path(file_path)

        if not path.is_file():
            print(f"File not found: {file_path}")
            return f"File not found: {file_path}"

        # Replace invalid UTF-8 characters so the tool can still read the file.
        content = path.read_text(encoding="utf-8", errors="replace")
        print("File read")
        return content if content.strip() else "The file is empty."

    except OSError as e:
        print(f"Error reading file: {e}")
        return f"Error reading file: {e}"

@tool
def read_csv_tool(file_path: str) -> str:
    """Read a CSV file and return its content.
    args:
        file_path (str): The path to the CSV file.
    returns:
        str: The content of the CSV file or a message if no content is found.
    """
    try:
        # Check for metadata (#)
        skiprows = 0

        with open(file_path, "r", encoding="utf-8-sig") as f:
            for line in f:
                if line.lstrip().startswith("#") or not line.strip():
                    skiprows += 1
                else:
                    break

        # Load dataframe
        df = pd.read_csv(file_path, skiprows=skiprows)
        print(f"Loaded CSV file of size {len(df)}, {len(df.columns)}")
        return f"Loaded CSV file of size {len(df)}, {len(df.columns)}. Content:\n{df.to_string(index=False)}"

    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return f"Error reading CSV file: {e}"

@tool
def read_excel_tool(file_path: str) -> str:
    """Read an Excel file and return its content.
    args:
        file_path (str): The path to the excel file.
    returns:
        str: The content of the excel file or a message if no content is found.
    """
    try:
        df = pd.read_excel(file_path)
        print(f"Loaded Excel file of size {len(df)}, {len(df.columns)}")
        return f"Loaded Excel file of size {len(df)}, {len(df.columns)}. Content:\n{df.to_string(index=False)}"
        
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return f"Error reading Excel file: {e}"

#################
## Image Tools ##
#################
import base64

@tool
def describe_image_assistant(image_path: str) -> str:
    """Describe the image, including any text, objects, or scenes present.

    Args:
        image_path (str): path to the image

    Returns:
        str: description of the image
    """
    try:
        path = Path(image_path)
        mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        image_data = base64.b64encode(path.read_bytes()).decode("utf-8")

        response = base_llm.invoke([
            HumanMessage(content=[
                {"type": "text", "text": "Describe the image, including any text, objects, or scenes present."},
                {
                    "type": "image",
                    "source_type": "base64",
                    "mime_type": mime_type,
                    "data": image_data,
                },
            ])
        ])
        print("Image analyzed")
        return response.content
    except Exception as e:
        print(f"Could not analyze the image: {e}")
        return f"Could not analyze the image: {e}"

@tool
def extract_text_from_image_assistant(image_path: str) -> str:
    """Extract text from the image

    Args:
        image_path (str): path to the image

    Returns:
        str: extracted text from the image
    """
    try:
        path = Path(image_path)
        mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        image_data = base64.b64encode(path.read_bytes()).decode("utf-8")

        response = base_llm.invoke([
            HumanMessage(content=[
                {"type": "text", "text": "Extract all readable text from this image."},
                {
                    "type": "image",
                    "source_type": "base64",
                    "mime_type": mime_type,
                    "data": image_data,
                },
            ])
        ])
        print("Image analyzed")
        return response.content
    except Exception as e:
        print(f"Could not analyze the image: {e}")
        return f"Could not analyze the image: {e}"

##############
## MP3 tool ##
##############
@tool
def analyze_mp3_tool(audio_path: str) -> str:
    """Analyze an MP3 and return a transcript or answer about its contents.
    
    Args:
        audio_path (str): path to the mp3
    
    Returns:
        str: extracted text from the mp3"""
    try:
        path = Path(audio_path)

        if not path.is_file():
            return f"Audio file not found: {audio_path}"

        audio_data = base64.b64encode(path.read_bytes()).decode("utf-8")

        response = base_llm.invoke([
            HumanMessage(content=[
                {
                    "type": "text",
                    "text": (
                        "Transcribe the spoken words in this audio verbatim. "
                        "Do not describe or summarize it. Return only the transcript."
                    ),
                },
                {
                    "type": "audio",
                    "base64": audio_data,
                    "mime_type": "audio/mpeg",
                },
            ])
        ])
        print("Image analyzed")
        return response.text
    except Exception as e:
        print(f"Could not analyze the MP3: {e}")
        return f"Could not analyze the MP3: {e}"