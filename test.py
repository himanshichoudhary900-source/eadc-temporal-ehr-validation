import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Use available model
model = genai.GenerativeModel('gemini-2.5-flash')

response = model.generate_content("Say hello and confirm you're working!")
print(response.text)
