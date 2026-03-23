import google.generativeai as genai

genai.configure(api_key="AIzaSyA4vOzZ6CFLvYdaMcEiIPSYuSgi3-6xTxs") 

# Fetch and print the available models
models = genai.list_models()

for model in models:
    print(model.name)  # Prints available model names
