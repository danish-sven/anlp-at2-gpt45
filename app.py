# Load the packages
import torch
import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel,BartTokenizer,BartForConditionalGeneration
import spacy
nlp=spacy.load("en_core_web_sm")
from spacy import displacy

#---Sidebar Design-----

st.sidebar.subheader("Select from the dropdown list") # add the subheader of sidebar
st.sidebar.text("") # add line space

option_lang = st.sidebar.selectbox(
    'What is your native language?',
    ('Japanese', 'Madarin'))   # add a dropdown list for native languages

st.sidebar.write('You selected:', option_lang)  # display the selected native language

st.sidebar.text("") # add line space


option_model=st.sidebar.selectbox(
    'Which language model would like to use?',
    ('GPT-2', 'BART'))  # add a dropdown list for language model

st.sidebar.write('You selected:', option_model)  # display the selected language model

#---Main Body Design-----

st.title('Make Friends with English 🤝')  # add a title for the web app

st.text("") # add line space

st.markdown('This web app is designed for ESL speakers who may face difficulty in communicating context in English.')
st.text("") # add line space

st.markdown('<p style="font-size:20px;"><strong>Enter your sentence 👇</strong></p>',unsafe_allow_html=True)  # add a subtitle

original = st.text_input('', '',label_visibility="collapsed") # add a textbox to input original sentence

go = st.button('Generate')   # add a 'Generate button' to run the selected language model

# Define the output directory
if option_model=='GPT-2':
    output_dir = "7. Models/"+'80K_GPT2_v2'+"/"

else:
    output_dir = "7. Models/"+'80K_BART_v2'+"/"


# Assign cuda to the device to use for training
if torch.cuda.is_available(): 
    dev = "cuda:0" 
    print("This model will run on CUDA")
# elif  torch.backends.mps.is_available(): 
#     dev = "mps:0"
#     print("This model will run on MPS")
else:
    dev = "cpu" 
    print("This model will run on CPU")
device = torch.device(dev) 


# Define the function to generate corrected sentence using GPT-2 model
def generate_prediction(prompt, max_length=100, temperature=1.0, top_p=1.0):
    model = GPT2LMHeadModel.from_pretrained(output_dir).to(device)
    tokenizer = GPT2Tokenizer.from_pretrained(output_dir)
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=device)
    with torch.no_grad():
        output = model.generate(
            input_ids, 
            attention_mask=attention_mask, 
            max_length=max_length, 
            num_return_sequences=1, 
            no_repeat_ngram_size=2,
            temperature=temperature,
            top_p=top_p,
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Define the function to extract the output (corrected sentence)
def model_running(model):
    if go and model=='GPT-2':
        try:
            tokenizer = GPT2Tokenizer.from_pretrained(output_dir)
            prompt = f"input: {original} output:"
            prompt_length = len(tokenizer.encode(prompt))
            dynamic_max_length = int(1.5 * len(original.split())) + prompt_length

            # Generate prediction
            prediction = generate_prediction(prompt, max_length=dynamic_max_length, temperature=0.8, top_p=0.8)

            # Extract the actual generated output
            generated_output = prediction.split("output:")[1].strip()

            return generated_output
        
        except Exception as e:
            st.exception("Exception: %s\n" % e)

    elif go and model=='BART':
        try:
            model = BartForConditionalGeneration.from_pretrained(output_dir)
            tokenizer = BartTokenizer.from_pretrained(output_dir)

            # Tokenize the input text
            input_ids = tokenizer.encode(original, return_tensors='pt')

            # Generate text with the fine-tuned BART model
            output_ids = model.generate(input_ids)

            # Decode the output text
            generated_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

            return generated_output

        except Exception as e:
            st.exception("Exception: %s\n" % e)


output=model_running(option_model)

# Add the warning message based on the output
if output is None:
    st.markdown('<span style="color: #FF4500;">Note: Please enter your sentence and click **Generate** button!</span>',unsafe_allow_html=True)
else:
    st.text("")

st.markdown('<p style="font-size:20px;"><strong>Recommended sentence 💡</strong></p>',unsafe_allow_html=True) # add a subtitle

st.text(output)  # display the corrected sentence

st.text("") # add line space

st.markdown('<p style="font-size:20px;"><strong>Part-of-speech Tagging 🏷</strong></p>',unsafe_allow_html=True)  # add a subtitle

# Add the POS tags
if original!='' and output is not None:
    doc=nlp(output)
    for token in doc:
        st.write(token,token.pos_)

st.text("") # add line space

st.markdown('<p style="font-size:20px;"><strong>Dependency Tree 🌳</strong></p>',unsafe_allow_html=True)  # add a subtitle

# Add a html wrapper to hold the html file of dependency tree
HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""

# Add the dependency tree
if original!='' and output is not None:
    doc=nlp(output)
    docs = [span.as_doc() for span in doc.sents]
    html=displacy.render(docs,style='dep')
    st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)


st.markdown('___')
st.markdown('by [A very beta ChatGPT-4.5](https://github.com/danish-sven/anlp-at2-gpt45/)')   # add the author


# # The code below is to generate corrected sentences with GPT-2 or BART model.
# if go and option_model=='GPT-2':
#     try:

#         model = GPT2LMHeadModel.from_pretrained(output_dir).to(device)
#         tokenizer = GPT2Tokenizer.from_pretrained(output_dir)

#         def generate_prediction(prompt, max_length=100, temperature=1.0, top_p=1.0):
#             input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
#             attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=device)
#             with torch.no_grad():
#                 output = model.generate(
#                     input_ids, 
#                     attention_mask=attention_mask, 
#                     max_length=max_length, 
#                     num_return_sequences=1, 
#                     no_repeat_ngram_size=2,
#                     temperature=temperature,
#                     top_p=top_p,
#                 )
#             return tokenizer.decode(output[0], skip_special_tokens=True)

#         # Set max_length dynamically based on the length of the original text
#         prompt = f"input: {original} output:"
#         prompt_length = len(tokenizer.encode(prompt))
#         dynamic_max_length = int(1.5 * len(original.split())) + prompt_length

#         # Generate prediction
#         prediction = generate_prediction(prompt, max_length=dynamic_max_length, temperature=0.8, top_p=0.8)

#         # Extract the actual generated output
#         generated_output = prediction.split("output:")[1].strip()

#         st.text(generated_output)

#     except Exception as e:
#         st.exception("Exception: %s\n" % e)

# elif go and option_model=='BART':
#     try:

#         model = BartForConditionalGeneration.from_pretrained(output_dir)
#         tokenizer = BartTokenizer.from_pretrained(output_dir)


#         # Tokenize the input text
#         input_ids = tokenizer.encode(original, return_tensors='pt')

#         # Generate text with the fine-tuned BART model
#         output_ids = model.generate(input_ids)

#         # Decode the output text
#         generated_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

#         st.text(generated_output)

#     except Exception as e:
#         st.exception("Exception: %s\n" % e)   




