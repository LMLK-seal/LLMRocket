import gradio as gr
import torch
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable OneDNN
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
import time
import fitz  # PyMuPDF for PDF processing



# Suppress TensorFlow messages (more targeted)
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR) 
logging.getLogger('transformers').setLevel(logging.ERROR)  
logging.getLogger('transformers.tokenization_utils_base').setLevel(logging.ERROR)
logging.getLogger('transformers.pipelines').setLevel(logging.ERROR)

# Set memory management for PyTorch
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# Constants
MAX_OUTPUT_TOKENS = 2048
MAX_IMAGE_SIZE = (1120, 1120)

# Default values for top_k and top_p
DEFAULT_TOP_K = 50
DEFAULT_TOP_P = 0.9

# Global variables
model = None
tokenizer = None
processor = None
is_vision_model = False

# Custom dark theme
dark_theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
    radius_size=gr.themes.sizes.radius_sm,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    body_background_fill="*neutral_950",
    body_text_color="*neutral_50",
    color_accent_soft="*primary_800",
    background_fill_secondary="*neutral_900",
    block_background_fill="*neutral_800",
    block_label_background_fill="*neutral_700",
    block_label_text_color="*neutral_200",
    button_primary_background_fill="*primary_700",
    button_primary_background_fill_hover="*primary_600",
    button_primary_text_color="white",
    button_secondary_background_fill="*neutral_700",
    button_secondary_background_fill_hover="*neutral_600",
    button_secondary_text_color="white",
    input_background_fill="*neutral_800",
    input_border_color="*neutral_700",
    input_placeholder_color="*neutral_400",
    slider_color="*primary_600",
)

def load_model(model_choice):
    global model, tokenizer, processor, is_vision_model
    
    if model_choice == "Llama-3.2-11B-Vision-Instruct":
        model_id = "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit"
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(model_id)
        is_vision_model = True

    elif model_choice == "Llama-3.2-3B-Instruct":
        model_id = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        is_vision_model = False

    elif model_choice == "Mistral-7B-Instruct-v0.2":
        model_id = "mistralai/Mistral-7B-Instruct-v0.2"
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        is_vision_model = False

    elif model_choice == "Llama-3.2-1B-Instruct":
        model_id = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        is_vision_model = False

    elif model_choice == "Phi-3-mini-4k-instruct":
        model_id = "unsloth/Phi-3-mini-4k-instruct"
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        is_vision_model = False

    elif model_choice == "gemma-2b-it":
        model_id = "unsloth/gemma-2b-it"
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        is_vision_model = False

    return f"Loaded {model_choice} successfully!"

def process_file(file):
    if file.name.endswith('.pdf'):
        text = ""
        with fitz.open(file.name) as doc:
            for page in doc:
                text += page.get_text()
        return text
    elif file.name.endswith('.txt'):
        with open(file.name, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        return "Unsupported file format. Please upload a PDF or TXT file."

def generate_response(image, user_prompt, temperature, max_tokens, history, file):
    global model, tokenizer, processor, is_vision_model
    
    if model is None:
        return history, "Please select a model first."

    start_time = time.time()
    
    if file:
        file_content = process_file(file)
        user_prompt = f"Analyze the following text:\n\n{file_content}\n\nUser query: {user_prompt}"

    if is_vision_model:
        if image is not None:
            image = image.resize(MAX_IMAGE_SIZE)
            prompt = f"<|image|><|begin_of_text|>{user_prompt} Answer:"
            inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
        else:
            return history, "Image is required for vision model."

        output = model.generate(
            **inputs,
            max_new_tokens=min(max_tokens, MAX_OUTPUT_TOKENS),
            temperature=temperature,
            top_k=DEFAULT_TOP_K,
            top_p=DEFAULT_TOP_P
        )

        raw_output = processor.decode(output[0])
        cleaned_output = raw_output.replace("<|image|><|begin_of_text|>", "").strip().replace(" Answer:", "")

    else:  # Text-only model
        if isinstance(model, type(AutoModelForCausalLM)):  # Llama-3.2-3B-Instruct
            prompt = f"[INST] {user_prompt} [/INST]"
        else:  # Mistral-7B-Instruct-v0.2
            prompt = f"<s>[INST] {user_prompt} [/INST]"

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

        output = model.generate(
            input_ids,
            max_new_tokens=min(max_tokens, MAX_OUTPUT_TOKENS),
            temperature=temperature,
            top_k=DEFAULT_TOP_K,
            top_p=DEFAULT_TOP_P,
            do_sample=True
        )

        cleaned_output = tokenizer.decode(output[0], skip_special_tokens=True)
        cleaned_output = cleaned_output.split("[/INST]")[-1].strip()

    if cleaned_output.startswith(user_prompt):
        cleaned_output = cleaned_output[len(user_prompt):].strip()
    
    end_time = time.time()
    tokens_generated = len(tokenizer.encode(cleaned_output))
    time_taken = end_time - start_time
    tokens_per_second = tokens_generated / time_taken

    cleaned_output += f"\n\nTokens/second: {tokens_per_second:.2f}"
    
    history.append((user_prompt, cleaned_output))
    return history, f"Generation complete. Tokens/second: {tokens_per_second:.2f}"

def clear_chat():
    return [], "Chat cleared."

def gradio_interface():
    with gr.Blocks(theme=dark_theme) as demo:
        gr.HTML(
        """
        <h1 style='text-align: center; color: #f0f0f0;'>
        LLMRocket powered by Gardio
        </h1>
        """)
        with gr.Row():
            with gr.Column(scale=1):
                model_choice = gr.Dropdown(
                    choices=["Llama-3.2-11B-Vision-Instruct", "Llama-3.2-3B-Instruct", "Mistral-7B-Instruct-v0.2", "Llama-3.2-1B-Instruct", "Phi-3-mini-4k-instruct", "gemma-2b-it"],
                    label="Select Model"
                )
                load_model_button = gr.Button("Load Model")
                model_status = gr.Textbox(label="Model Status", interactive=False)
                
                with gr.Row():
                    image_input = gr.Image(
                        label="Image", 
                        type="pil", 
                        image_mode="RGB", 
                        height=512,
                        width=512
                    )
                    file_input = gr.File(label="Upload PDF/TXT")

                temperature = gr.Slider(
                    label="Temperature", minimum=0.1, maximum=2.0, value=0.6, step=0.1)
                max_tokens = gr.Slider(
                    label="Max Tokens", minimum=50, maximum=MAX_OUTPUT_TOKENS, value=1000, step=50)

            with gr.Column(scale=2):
                chat_history = gr.Chatbot(label="Chat", height=700)
                generation_info = gr.Textbox(label="Generation Info", interactive=False)

                user_prompt = gr.Textbox(
                    show_label=False,
                    container=False,
                    placeholder="Enter your prompt", 
                    lines=2
                )

                with gr.Row():
                    generate_button = gr.Button("Generate")
                    clear_button = gr.Button("Clear")

        load_model_button.click(
            fn=load_model,
            inputs=[model_choice],
            outputs=[model_status]
        )

        generate_button.click(
            fn=generate_response, 
            inputs=[image_input, user_prompt, temperature, max_tokens, chat_history, file_input],
            outputs=[chat_history, generation_info]
        )

        clear_button.click(
            fn=clear_chat,
            inputs=[],
            outputs=[chat_history, generation_info]
        )

    return demo

demo = gradio_interface()
demo.launch()
