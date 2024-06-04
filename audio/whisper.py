
# ffmpeg -i Test.m4a output.wav

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset

cuda_available = torch.cuda.is_available()
cuda_available = False
device = "cuda:0" if cuda_available else "cpu"
torch_dtype = torch.float16 if cuda_available else torch.float32

# model_id = "openai/whisper-large-v3"
# model_id = "openai/whisper-medium"
model_id = "openai/whisper-tiny"

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
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

# dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
# sample = dataset[0]["audio"]
# # sample.to(device)

# ffmpeg -i Test.m4a output.wav
file_name = "audio_test_files/test2.wav"
# result = pipe(file_name)
result = pipe(file_name, generate_kwargs={"language": "serbian"})
print(result["text"])