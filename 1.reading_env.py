from decouple import config

OPENAI_API_KEY = config('OPENAI_API_KEY')

print(OPENAI_API_KEY)