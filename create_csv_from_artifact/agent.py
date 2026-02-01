import os
import io
import pypdf
from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools import ToolContext

load_dotenv()

# --- Configuration ---
V_LINES = {
    "v1": os.getenv("VOICE_LINE1"),
    "v2": os.getenv("VOICE_LINE2"),
    "v3": os.getenv("VOICE_LINE3"),
    "v4": os.getenv("VOICE_LINE4"),
}

# --- Unified Instructions ---
BILLING_ASSISTANT_INSTRUCTIONS = f"""
You are a Billing Specialist. Your job is to process PDF bills and extract data into a CSV.

FOLLOW THESE STEPS:
1. When a user uploads a PDF, FIRST use the 'get_pdf_text_from_artifact' tool to read it.
2. Locate the 'DETAILED CHARGES' and 'PLANS' sections.
3. Extract 'base_charge' and extra charges for these lines: {list(V_LINES.values())}.
4. Logic:
   - 'Kickback discount shown above' -> charge = 15
   - 'Includes $5.00 AutoPay and $10.00 Kickback' -> charge = 15
   - 'Use less than 2.0GB...' -> charge = 25
   - 'Includes $5.00 AutoPay' -> charge = 5
5. Generate a CSV string of the results.
6. Use the 'save_csv_artifact' tool to save the final 'extracted_charges.csv'.
"""

# --- Tools ---

async def get_pdf_text_from_artifact(tool_context: ToolContext) -> str:
    """Finds a PDF in the session and extracts its text content."""
    artifacts = await tool_context.list_artifacts()
    # Find the first PDF artifact
    pdf_art = next((a for a in artifacts if "pdf" in a.name.lower() or (hasattr(a, 'mime_type') and (a.mime_type == "application/pdf" or a.mime_type == "application/octet-stream"))), None)
    
    if not pdf_art:
        return "Error: No PDF found. Please upload the bill."

    full_art = await tool_context.load_artifact(filename=pdf_art.name)
    
    if full_art.inline_data:
        pdf_file = io.BytesIO(full_art.inline_data.data)
        reader = pypdf.PdfReader(pdf_file)
        return "\n".join([page.extract_text() for page in reader.pages])
    return "Error: Could not read PDF data."

async def save_csv_artifact(tool_context: ToolContext, csv_data: str, filename: str = "extracted_charges.csv") -> str:
    """Saves the generated CSV string as a session artifact."""
    from google.genai import types
    data_bytes = csv_data.encode('utf-8')
    artifact_part = types.Part(inline_data=types.Blob(mime_type="text/csv", data=data_bytes))
    await tool_context.save_artifact(filename, artifact_part)
    return f"Successfully saved {filename} to artifacts."

# --- Agent Definition ---

root_agent = LlmAgent(
    name="billing_specialist",
    model="gemini-2.5-flash-lite", # Use a standard stable model name for better compatibility
    instruction=BILLING_ASSISTANT_INSTRUCTIONS,
    tools=[get_pdf_text_from_artifact, save_csv_artifact],
)

