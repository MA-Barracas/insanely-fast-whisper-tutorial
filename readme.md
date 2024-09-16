## **Mastering YouTube Audio Transcription with Python: A Step-by-Step Guide Using Huggingface and Insanely-Fast-Whisper**

In the rapidly evolving world of data science and AI, the ability to efficiently process and analyze audio data can open doors to countless opportunities. Whether it's for accessibility, content creation, or data analysis, transcribing audio into text is a crucial skill. But beyond just getting the job done, doing it quickly and without spending a dime adds significant value. This tutorial is designed to show you how to download, transcribe, and summarize audio from YouTube videos using Python, all while minimizing costs and maximizing speed.

We'll explore two powerful Python libraries: Huggingface Transformers and insanely-fast-whisper. Huggingface’s Whisper model offers high accuracy, but when time is of the essence, insanely-fast-whisper provides a lightning-fast alternative. Best of all, the entire process can be done for free using Google Colab with a T4 GPU, making it accessible to anyone with a GPU or a Mac.

### **Step 1: Setting Up the Environment**

Before diving into the code, let's ensure our environment is ready. We'll be using a Google Colab notebook for this tutorial, which offers free access to a T4 GPU, perfect for handling the demands of audio processing. The first step is to install the necessary Python libraries.

```python
!pip install --upgrade pip
!pip install --upgrade transformers datasets[audio] accelerate
!pip install --upgrade pytubefix pydub
```

**Explanation:**

1. **`pip install --upgrade pip`**: Keeping `pip` updated ensures you have access to the latest versions of libraries, which can include important bug fixes and performance improvements.
   
2. **`pip install --upgrade transformers datasets[audio] accelerate`**: 
   - **Transformers**: This library by Huggingface provides a vast collection of pre-trained models for various NLP tasks, including speech recognition.
   - **Datasets[audio]**: This extension of the Huggingface Datasets library is essential for loading and preprocessing audio datasets.
   - **Accelerate**: This library optimizes model performance, particularly on GPUs, ensuring that your transcription process is as efficient as possible.
   
3. **`pip install --upgrade pytubefix pydub`**:
   - **Pytubefix**: Due to recent bugs in `pytube`, we use `pytubefix` to reliably download YouTube videos. This is crucial for ensuring the audio is correctly extracted.
   - **Pydub**: A library for audio processing, `pydub` handles conversions between different audio formats, which is a key step in preparing the audio for transcription.

### **Step 2: Downloading and Converting YouTube Audio to MP3**

With the environment set up, the next step is to download the audio from a YouTube video and convert it to an MP3 file. This process is essential because most speech-to-text models, including the ones we’ll be using, work best with common audio formats like MP3 or WAV.

```python
from pytubefix import YouTube  
from pydub import AudioSegment
import os
import traceback  

def download_audio_mp3(url):
    try:
        yt = YouTube(url)
        video = yt.streams.filter(only_audio=True).first()
        output_file = video.download()

        mp3_filename = os.path.splitext(output_file)[0] + ".mp3"
        audio = AudioSegment.from_file(output_file)
        audio.export(mp3_filename, format="mp3")

        os.remove(output_file)

        print(f"MP3 saved as: {mp3_filename}")
        return mp3_filename
    except Exception as e:
        print("Error: ", e)
        print(traceback.format_exc())
        return None
```

**Explanation:**

- **`YouTube(url)`**: Creates an instance of the YouTube object using the provided video URL.
- **`video = yt.streams.filter(only_audio=True).first()`**: Filters out all non-audio streams, ensuring we download only the audio.
- **`video.download()`**: Downloads the selected audio stream.
- **`AudioSegment.from_file(output_file)`**: Reads the downloaded audio file, allowing further manipulation.
- **`audio.export(mp3_filename, format="mp3")`**: Converts the downloaded audio into MP3 format, which is compatible with our transcription models.
- **`os.remove(output_file)`**: Deletes the original file to save space, since it’s no longer needed after conversion.
- **`traceback.format_exc()`**: Provides detailed error information if something goes wrong, making debugging easier.

This function ensures that the process of downloading and converting audio is both efficient and resilient to errors.

### **Step 3: Transcribing Audio with Huggingface Transformers**

With our MP3 file ready, the next step is to transcribe the audio into text. We’ll use Huggingface’s `whisper-large-v3` model, known for its accuracy in automatic speech recognition (ASR). This model requires around 3GB of GPU memory, which the free T4 GPU in Colab handles well.

```python
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

result = pipe(audio_filename)
```

**Explanation:**

- **`device = "cuda:0" if torch.cuda.is_available() else "cpu"`**: Automatically selects the best available hardware (GPU or CPU) for processing. Utilizing a GPU significantly speeds up the transcription process.
- **`torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32`**: This sets the data type to `float16` on GPUs to reduce memory usage, which is critical when working with large models.
- **`AutoModelForSpeechSeq2Seq.from_pretrained(model_id, ...)`**: Loads the Whisper model pre-trained on speech-to-text tasks. It’s optimized for low memory usage while still providing high accuracy.
- **`AutoProcessor.from_pretrained(model_id)`**: The processor handles the necessary preprocessing steps, such as tokenization and feature extraction, ensuring the audio is in the correct format for the model.
- **`pipeline(...)`**: Simplifies the inference process by combining the model and processor into a single pipeline for automatic speech recognition.
- **`result = pipe(audio_filename)`**: This line performs the actual transcription, converting the audio into text. The process takes around five minutes for a 40-minute audio file.

### **Step 4: Summarizing the Transcription with Cohere API**

Once we have the transcription, summarizing the text can help extract key insights quickly. The Cohere API is perfect for this, offering powerful NLP capabilities that allow us to generate structured summaries from large volumes of text.

```python
import cohere
from google.colab import userdata

co = cohere.Client(api_key=userdata.get('COHERE_API_KEY'))

query = f"""
Write a thorough summary for this text: '''
{result["text"]}
'''
Give extensive info about the content. 
Discuss all important points raised throughout.
At the end, create a section of "Conclusions" and another for "Summary"
"""

cohere_query = co.chat(
  model="command-r-plus",
  message=query
)

from IPython.display import Markdown
display(Markdown(cohere_query.text))
```

**Explanation:**

- **`cohere.Client(api_key=userdata.get('COHERE_API_KEY'))`**: Initializes the Cohere API client using your API key, enabling access to its text generation and summarization capabilities.
- **`query = f"""Write a thorough summary for this text: ''' {result["text"]} '''"""`**: This query prompts the Cohere model to generate a detailed summary, helping to distill the most important points from the transcription.
- **`co.chat(...)`**: Sends the query to the Cohere model and retrieves the summarized text.
- **`display(Markdown(cohere_query.text))`**: Displays the summary in a readable Markdown format within the Colab notebook.

### **Step 5: Transcribing Audio with Insanely-Fast-Whisper**

While Huggingface’s Whisper model is highly accurate, sometimes speed is of the essence. Insanely-fast-whisper provides a faster alternative, transcribing audio files in a fraction of the time. Let’s install and use it:

```python
!pip install insanely-fast-whisper
```

**Explanation:**

- **`insanely-fast-whisper`**: This library is specifically optimized for speed, allowing you to transcribe audio quickly, making it ideal for scenarios where time is a critical factor.

Next, we need to install `pipx` and `venv` to run the CLI-based insanely-fast-whisper:

```python
!pip install -q pipx && apt install python3.10-venv
```

**Explanation:**

- **`pipx`**: This tool installs and runs Python applications in isolated environments, preventing dependency conflicts.
- **`python3

.10-venv`**: The `venv` module creates isolated Python environments, ensuring that different projects don’t interfere with each other.

Now, let’s run insanely-fast-whisper:

```python
!pipx run insanely-fast-whisper --file-name "Cursor AI tutorial for beginners.mp3"
```

**Explanation:**

- **`!pipx run insanely-fast-whisper --file-name "Cursor AI tutorial for beginners.mp3"`**: This command uses `pipx` to run insanely-fast-whisper, which transcribes the audio file in about 1 minute and 40 seconds—much faster than the 5 minutes required by Huggingface’s model. However, it does require more GPU memory, around 9GB, so it’s essential to ensure your environment can handle this.

### **Conclusion**

This tutorial has guided you through the process of downloading, transcribing, and summarizing audio from YouTube videos using Python, all while keeping costs at zero and maximizing speed. By leveraging Huggingface Transformers and insanely-fast-whisper, you can choose the tool that best suits your needs—whether it’s the high accuracy of Whisper or the blazing speed of insanely-fast-whisper. Combined with the Cohere API for summarization, this workflow provides a powerful, efficient way to handle audio data.

With these tools, you’re not just transcribing audio—you’re doing it faster, for free, and in a way that can easily be integrated into various workflows, from content creation to data analysis. Try it out and see how it can transform your approach to working with audio content!

