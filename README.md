# LLMRocket Powered by Gradio.

![LLMRocket](https://github.com/LMLK-seal/LLMRocket/blob/main/Img.png?raw=true)

This repository contains a Gradio-based interface for interacting with various large language models, including vision-language models. The application allows users to select different models, process images and text inputs, and generate responses based on the chosen model.
Temporary public link can be generated that you can use to access the interface from any device (Valid for 72 huors) - See description for more info.

Important Note: LLMRocket is designed to be 100% offline and private once set up. However, it requires an initial download of the model. After this initial download, the application operates entirely on your local machine, ensuring your prompts remain private and secure.

## 🚀 Features

- Support for multiple language models, including vision-language models
- Image input for vision models
- PDF and TXT file processing
- Adjustable generation parameters (temperature, max tokens)
- Dark theme UI
- Performance metrics (tokens per second)

## 📦 Installation

1. Clone the repository or download the `Main.py` file.

2. Install the required libraries:

```bash
pip install gradio torch Pillow transformers PyMuPDF
```

## 🖥️ Usage

1. Run the script:

```bash
python Main.py
```

2. Open your web browser and navigate to the URL provided in the console output (usually `http://localhost:7860`) to view the application.

3. Using the application:
   1. Select a model from the dropdown menu and click "Load Model".
   2. For vision models, upload an image using the image input area.
   3. For text processing, you can upload a PDF or TXT file.
   4. Enter your prompt in the text box.
   5. Adjust the temperature and max tokens sliders if desired.
   6. Click "Generate" to get a response from the model.
   7. The chat history will display your prompts and the model's responses.
   8. Use the "Clear" button to reset the chat history.

4. To create a temporary public URL for remote access (valid for 72 hours), add `share=True` to the `demo.launch()` function at the end of the script:

```python
demo.launch(share=True)
```

This will generate a public URL that you can use to access the interface from any device.

## 🤖 Supported Models

- Llama-3.2-11B-Vision-Instruct
- Llama-3.2-3B-Instruct
- Mistral-7B-Instruct-v0.2
- Llama-3.2-1B-Instruct
- Phi-3-mini-4k-instruct
- gemma-2b-it

## ⚠️ Note

Ensure you have sufficient GPU memory to run the larger models. Adjust the `MAX_OUTPUT_TOKENS` constant if needed to limit token generation.

## 📄 License

This project is licensed under the MIT License.

## 🤝 Contributing

Contributions to this project are welcome!

## 🙏 Acknowledgments

- This project uses models from Hugging Face's Transformers library.
- UI is built using the Gradio library.
