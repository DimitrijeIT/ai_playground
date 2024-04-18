# pip install pyaudio

import pyaudio
import numpy as np
from transformers import WhisperForConditionalGeneration, WhisperTokenizer

# Initialize the Whisper model and tokenizer
model_name = "openai/whisper-large"
tokenizer = WhisperTokenizer.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)

# Setup audio stream using PyAudio
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=4096)
stream.start_stream()

def process_audio(audio_chunk):
    inputs = tokenizer(audio_chunk.numpy(), return_tensors="pt")
    with torch.no_grad():
        predicted_ids = model.generate(inputs.input_values, max_length=100)
    transcription = tokenizer.batch_decode(predicted_ids)[0]
    return transcription

# Main loop for real-time transcription
try:
    while True:
        audio_chunk = np.frombuffer(stream.read(4096), dtype=np.int16)
        transcription = process_audio(audio_chunk)
        print(transcription)
except KeyboardInterrupt:
    # Stop and close the stream and PyAudio
    stream.stop_stream()
    stream.close()
    p.terminate()
