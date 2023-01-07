import openai

openai.api_key = API_KEY

try:
    print(openai.Model.list())
except Exception as e:
    print(f"Error: {e}")

prompt = """Decide whether a Tweet's sentiment is positive, neutral, or negative.

Tweet: I dont like the new Batman movie!
Sentiment:"""
response = openai.Completion.create(model="text-davinci-003", prompt= prompt, max_tokens=6)

print(response)