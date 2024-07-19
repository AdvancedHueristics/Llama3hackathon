from together import Together

def get_env_data_as_dict(path: str) -> dict:
    with open(path, 'r') as f:
       return dict(tuple(line.replace('\n', '').split('=')) for line
                in f.readlines() if not line.startswith('#'))

env_dict = get_env_data_as_dict('.env')
client = Together(api_key=env_dict.get("TOGETHER_API_KEY", ""))

stream = client.chat.completions.create(
  model="meta-llama/Meta-Llama-3-8B-Instruct-Turbo",
  messages=[{"role": "user", "content": "What are some fun things to do in New York?"}],
  stream=True,
)

for chunk in stream:
  print(chunk.choices[0].delta.content or "", end="", flush=True)