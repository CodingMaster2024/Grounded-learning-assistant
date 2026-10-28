# Grounded Learning Assistant — Cloud Build

Streamlit app using Groq-hosted chat completion API. This build does not require Ollama or a local embedding model. Document retrieval uses lightweight keyword matching.

## Deploy
1. Upload these files to a GitHub repository.
2. In Streamlit Community Cloud, create an app from that repository and set the main file to `app.py`.
3. In the deployed app's Settings → Secrets, add:

```toml
GROQ_API_KEY = "paste-your-key-here"
```

Never commit your API key to GitHub. Create a key at https://console.groq.com/ . The provider's free usage limits and model availability may change.

## Limitations
- Cloud AI answers can be wrong; it cannot guarantee flawless answers.
- Uploaded documents are processed during the session and app storage may be temporary.
- Browser tab monitoring is disabled because the original monitor only worked on localhost.
- Keyword retrieval is lighter but less semantically capable than embedding-based retrieval.
