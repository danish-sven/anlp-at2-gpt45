import torch
import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel,BartTokenizer,BartForConditionalGeneration

st.title('Learn English with Me ')
st.markdown("A [simple demonstration](https://github.com/CaliberAI/streamlit-get-stories-aylien) of using [Streamlit](https://streamlit.io/) with [HuggingFace's GPT-2](https://github.com/huggingface/transformers/).")

option_lang = st.selectbox(
    'What is your native language?',
    ('Japanese', 'Madarin'))

st.write('You selected:', option_lang)

st.text("") # add line space


original = st.text_input('Enter your sentence 👇', 'This is a placeholder')
# num_return_sequences = st.number_input('Number of generated sequences', 1, 100, 20)
# max_length = st.number_input('Length of sequences', 5, 100, 20)

st.text("") # add line space

option_model=st.selectbox(
    'Which language model would like to use?',
    ('GPT-2', 'BART'))

if option_model=='GPT-2':
    output_dir = "7. Models/"+'80K_GPT2_v2'+"/"

else:
    output_dir = "7. Models/"+'80K_BART_v2'+"/"

go = st.button('Generate')


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


# The code below needs to be replaced with GPT-2 model or BART model.
if go and option_model=='GPT-2':
    try:

        model = GPT2LMHeadModel.from_pretrained(output_dir).to(device)
        tokenizer = GPT2Tokenizer.from_pretrained(output_dir)

        def generate_prediction(prompt, max_length=100, temperature=1.0, top_p=1.0):
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

        # Set max_length dynamically based on the length of the original text
        prompt = f"input: {original} output:"
        prompt_length = len(tokenizer.encode(prompt))
        dynamic_max_length = int(1.5 * len(original.split())) + prompt_length

        # Generate prediction
        prediction = generate_prediction(prompt, max_length=dynamic_max_length, temperature=0.8, top_p=0.8)

        # Extract the actual generated output
        generated_output = prediction.split("output:")[1].strip()

        st.text(generated_output)

    except Exception as e:
        st.exception("Exception: %s\n" % e)

elif go and option_model=='BART':
    try:

        model = BartForConditionalGeneration.from_pretrained(output_dir)
        tokenizer = BartTokenizer.from_pretrained(output_dir)


        # Tokenize the input text
        input_ids = tokenizer.encode(original, return_tensors='pt')

        # Generate text with the fine-tuned BART model
        output_ids = model.generate(input_ids)

        # Decode the output text
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        st.text(output_text)

    except Exception as e:
        st.exception("Exception: %s\n" % e)   


st.markdown('___')
st.markdown('by [A very beta ChatGPT-4.5](https://github.com/danish-sven/anlp-at2-gpt45/)')

