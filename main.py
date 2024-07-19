from together import Together
import os

# Load environment variables
def load_env_vars(path: str) -> dict:
    env_vars = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                key, value = line.split('=', 1)
                env_vars[key] = value
    return env_vars

# Initialize NLP client
env_vars = load_env_vars('sample.env')
api_key = env_vars.get("TOGETHER_API_KEY", "")
if not api_key:
    raise ValueError("API key is missing")

client = Together(api_key=api_key)

def get_response(user_input: str) -> str:
    try:
        stream = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3-8B-Instruct-Turbo",
            messages=[{"role": "user", "content": user_input}],
            stream=True,
        )
        response = ""
        for chunk in stream:
            response += chunk.choices[0].delta.content or ""
        return response
    except Exception as e:
        return f"An error occurred: {e}"
