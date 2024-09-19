
### **YouTube Audio Transcription with Python: A Step-by-Step Guide Using Insanely-Fast-Whisper**

Processing and analyzing audio data can be a game-changer for many fields, whether you're a researcher, content creator, or data scientist. Transcribing YouTube videos into text allows for better accessibility, analysis, and repurposing of content. But what if you could do it very quickly and at zero cost? In this tutorial, we'll walk through how to download, transcribe, and summarize audio from YouTube videos using Python, utilizing tools like Huggingface's Transformers and insanely-fast-whisper, all for free and fast.

### **Why Speed and Cost Matter**

Transcribing audio is crucial, but being able to do it efficiently and without spending any money adds an extra layer of utility. If you need to process large batches of audio or transcribe lengthy videos, speed can save you hours, and the ability to do it on free platforms like Google Colab makes this process accessible to everyone, not just those with deep pockets.

### **Step 1: Setting Up the Environment**

Let’s start by setting up the necessary environment. We'll be using a Google Colab notebook, which provides free access to a T4 GPU, making it ideal for handling audio transcription tasks. The first step is to install the required libraries.

```python
!pip install --upgrade pip
!pip install --upgrade transformers accelerate
!pip install --upgrade pytubefix pydub
```

**Explanation:**

1. **`pip install --upgrade pip`**: Keeping `pip` up to date ensures compatibility with the latest libraries and avoids potential installation issues.
   
2. **`pip install --upgrade transformers accelerate`**:
   - **Transformers**: This library by Huggingface contains cutting-edge models for a variety of NLP tasks, including automatic speech recognition (ASR).
   - **Accelerate**: This library optimizes the performance of the Huggingface models, particularly when running on GPUs, helping us make the most of the available hardware resources.
   
3. **`pip install --upgrade pytubefix pydub`**:
   - **Pytubefix**: This is a fork of `pytube`, a YouTube downloader library. `pytubefix` resolves recent bugs, ensuring you can download YouTube audio streams without interruptions.
   - **Pydub**: A library for handling and converting audio formats, `pydub` is essential for converting downloaded audio to MP3 format, making it compatible with our speech recognition models.

### **Step 2: Downloading and Converting YouTube Audio to MP3**

Now that the environment is set up, the next step is to download the audio from a YouTube video and convert it to an MP3 file. We’ll be using Geoffrey Hinton’s lecture, “Will Digital Intelligence Replace Biological Intelligence?”.

```python
from pytubefix import YouTube  
from pydub import AudioSegment
import os
import traceback 

def descargar_audio_mp3(url):
    try:
        # Descarga del video
        yt = YouTube(url)
        video = yt.streams.filter(only_audio=True).first()
        output_file = video.download()

        # Conversión a MP3 usando pydub
        mp3_filename = os.path.splitext(output_file)[0] + ".mp3"
        audio = AudioSegment.from_file(output_file)
        audio.export(mp3_filename, format="mp3")

        # Eliminación del archivo original
        os.remove(output_file)

        print(f"Archivo MP3 guardado como: {mp3_filename}")
        return mp3_filename
    except Exception as e:
        print("Ha ocurrido un error: ", e)
        print(traceback.format_exc())
        return None

url = "https://www.youtube.com/watch?v=N1TEjTeQeg0"
audio_filename = descargar_audio_mp3(url)
```

**Explanation:**

- **`YouTube(url)`**: Initializes a YouTube object with the video URL.
- **`yt.streams.filter(only_audio=True).first()`**: Filters the available streams to download only the audio, reducing unnecessary file size.
- **`video.download()`**: Downloads the audio stream to the local environment.
- **`AudioSegment.from_file(output_file)`**: Uses `pydub` to load the downloaded audio.
- **`audio.export(mp3_filename, format="mp3")`**: Converts the audio file to MP3 format, which is compatible with most speech recognition models.
- **`os.remove(output_file)`**: Deletes the original audio file after conversion to save space.

This function ensures a seamless process of downloading and converting audio into a format ready for transcription.

### **Step 3: Transcribing Audio with Huggingface Transformers**

With the MP3 file ready, we can now use Huggingface’s `whisper-large-v3` model to transcribe the audio into text. This model is known for its high accuracy in automatic speech recognition tasks.

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
    batch_size=24
)

result = pipe(audio_filename)
```

**Explanation:**

- **`device = "cuda:0" if torch.cuda.is_available() else "cpu"`**: Automatically detects whether a GPU is available and uses it if possible, as GPUs significantly speed up the transcription process.
- **`torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32`**: Chooses the most appropriate data type based on the hardware, with `float16` optimized for GPUs to reduce memory usage without sacrificing accuracy.
- **`AutoModelForSpeechSeq2Seq.from_pretrained(model_id, ...)`**: Loads the pre-trained Whisper model optimized for speech recognition.
- **`pipeline(...)`**: Combines the model and processor into a single pipeline that simplifies the transcription process.
- **`batch_size=24`**: Increases the batch size, which helps with processing speed when working with longer files.

This transcription process takes a few minutes depending on the length of the audio file and the hardware available.

### **Step 4: Summarizing the Transcription with Cohere API**

Once we have the transcription, the next step is to summarize the content using the Cohere API. This is useful for quickly extracting key points from a long transcription.

```python
import cohere
from google.colab import userdata
import os

# Check if running in Colab
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# Get API key
if IN_COLAB:
    api_key = userdata.get('COHERE_API_KEY')
else:
    api_key = os.environ.get('COHERE_API_KEY')

co = cohere.Client(api_key=api_key)

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

- **`userdata.get('COHERE_API_KEY')`**: Fetches the Cohere API key in Google Colab, or from the environment if running locally.
- **`cohere.Client(api_key)`**: Initializes the Cohere client using the provided API key.
- **`co.chat(model="command-r-plus", message=query)`**: Sends the transcription to Cohere for summarization using the `command-r-plus` model, which generates detailed summaries of long texts.

This step is particularly useful when dealing with lengthy videos, saving time by automatically creating a concise overview of the transcription.

### **Step 5: Transcribing Audio with Insanely-Fast-Whisper**

If time is a concern, we can use the insanely-fast-whisper library, which dramatically speeds up the transcription process. This tool is ideal when speed is prioritized over detailed control.

```python
!pip install insanely-fast-whisper
```

Once installed, we run the transcription process through the command line interface (CLI) using `pipx`:

```python
!pip install -q pipx && apt install python3.10-venv
!pipx run insanely-fast-whisper --file-name "Cursor AI tutorial for beginners.mp3"
```

**Explanation:**

- **`pipx`**: Allows running Python applications in isolated environments without installing them globally, reducing the risk of conflicts with other packages.
- **`python3.10-venv`**: Sets up a virtual environment, ensuring that dependencies do not interfere with other projects.
- **`pipx run insanely-fast-whisper`**: Runs the transcription directly from the command line, offering faster results than Huggingface Transformers, albeit with higher GPU memory usage (around 9GB).

This transcription process takes just over a minute,

 making it perfect for scenarios where time is limited.

### **Conclusion**

In this tutorial, we explored how to download, transcribe, and summarize YouTube videos using Python. By leveraging the Huggingface Transformers and insanely-fast-whisper libraries, you can choose between high accuracy or speed, depending on your needs. The entire process, from downloading to summarizing, can be done for free in Google Colab, making it accessible to anyone with basic Python knowledge and access to a GPU.

This workflow provides an efficient way to handle audio data, whether you need detailed transcripts or quick summaries. Try it out and see how it can transform your approach to working with audio content!

