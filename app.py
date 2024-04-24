import gradio as gr
import torch
from transformers import pipeline
import librosa

# モデルの設定
model_id = "kotoba-tech/kotoba-whisper-v1.0"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_kwargs = {"attn_implementation": "sdpa"} if torch.cuda.is_available() else {}
generate_kwargs = {"language": "japanese", "task": "transcribe"}

# モデルのロード
pipe = pipeline(
    "automatic-speech-recognition",
    model=model_id,
    torch_dtype=torch_dtype,
    device=device,
    model_kwargs=model_kwargs
)

# 文字起こし関数
def transcribe(audio_file):
    # 音声の読み込み
    audio, sr = librosa.load(audio_file, sr=None)
    
    # 音声をリサンプリング
    target_sr = 16000
    audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    
    # 推論の実行
    result = pipe(audio_resampled, generate_kwargs=generate_kwargs)
    
    return result["text"]

description = """
The bot was trained to answer questions based on Rick and Morty dialogues. Ask Rick anything!
<img src="https://huggingface.co/spaces/course-demos/Rick_and_Morty_QA/resolve/main/rick.png" width=200px>
"""

# Gradioインターフェースの定義
iface = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(type="filepath", label="Upload Audio (MP3 or MP4)"),
    outputs="text",
    title="Speech-to-Text App",
    description=description,
    theme=gr.themes.Soft(),
)
# アプリの起動
iface.launch(server_name="0.0.0.0", server_port=7860, share=True)