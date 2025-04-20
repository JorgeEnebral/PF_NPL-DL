import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

api_key = os.getenv("key")

client = Groq(api_key=api_key)


def generate_response(message: str) -> str:
    completion = client.chat.completions.create(
        model="deepseek-r1-distill-llama-70b",
        messages=[{"role": "user", "content": message}],
        temperature=0.6,
        max_completion_tokens=1000,
        top_p=0.95,
        stream=True,
        stop=None,
    )

    response = ""
    for chunk in completion:
        content = chunk.choices[0].delta.content
        if content:
            response += content
    return response
