# ANLP AT2 Web App - Make Friends with English

## Project Description
This project is aimed at developing a web application that takes sentences input by users and generate corrected sentences using fine-tuned language models including GPT-2 and BART. Part-of-speech tagging and dependency tree are created to provide visualisations on the output to further aid the user's learning.

## Team Name
A Very Beta ChatGPT 4.5

## Team Members
- Stefan Hall
- Tim Wang
- Amy Yang

## Development
To run locally:
- `pip install -r requirements.txt`
- `streamlit run app.py`

## Deployment
This app works well with Hugging Face Space. 
- Fine-tuned models are pushed to [Models](https://huggingface.co/amyyang) on Hugging Face.
- Find our web app [**here**](https://huggingface.co/spaces/amyyang/webapp_englishtool)


## References
- [BART Model](https://huggingface.co/docs/transformers/main/en/model_doc/bart)
- [CaliberAI/streamlit-nlg-gpt-2](https://github.com/CaliberAI/streamlit-nlg-gpt-2)
- [Video: Deploy Fine Tuned BERT or Transformers model on Streamlit Cloud](https://www.youtube.com/watch?v=mvIp9TvPMh0&t=348s)
- [OpenAI GPT2](https://huggingface.co/docs/transformers/main/en/model_doc/gpt2#overview)
- Gupta, S. (2020, December 10). Parts-of-Speech tagging app using Streamlit, spacy and Python. Srishti Gupta's Blog. https://srishti.hashnode.dev/parts-of-speech-tagging-app-using-streamlit-spacy-and-python
- [NAIST Lang-8 Learner Corpora](https://sites.google.com/site/naistlang8corpora/home?authuser=0)
