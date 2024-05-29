# How to start

1. Prepare a virtual environment: `pyenv virtual 3.11.3 pdf_chatbot_env; pyenv activate pdf_chatbot_env`
2. Prepare all the dependency: `pip install -f requirements.txt`
3. The app needs a web server to host pdf file for citation, so start pdf server first: `cd static; python -m http.server 8900`
4. Run the app: `streamlit run pdf_chatbot.py`

Note: openai apikey is required: `export OPENAI_API_KEY=sk-xxxxx`