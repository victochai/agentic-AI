import openai
import os
from dotenv import load_dotenv
load_dotenv()

api_key=os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")

client = openai.OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
    )

with open("instructions.txt", "r", encoding="utf-8") as f:
    instructions = f.read()

response = client.responses.create(
    model="gpt-4.1",
    instructions=instructions, # role --> developer
    input="I have basel, mozarella, rigatoni, parmiggiano, duck" # role --> user
)

print(response.output_text)
