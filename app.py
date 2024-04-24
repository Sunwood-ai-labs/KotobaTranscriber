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
<p align="center">
<img src="https://media.githubusercontent.com/media/Sunwood-ai-labs/KotobaTranscriber/main/docs/ship2.gif" width="70%">
<br>
</p>
"""

theme = gr.themes.Soft(
    # neutral_hue=gr.themes.Color(c100="#f3f4f6", c200="#e5e7eb", c300="#d1d5db", c400="#9ca3af", c50="#ecf1e8", c500="#6b7280", c600="#4b5563", c700="#374151", c800="#1f2937", c900="#1E2D2F", c950="#1E2D2F"),
    primary_hue="gray",
    neutral_hue=gr.themes.Color(c100="#f3f4f6", c200="#e5e7eb", c300="#d1d5db", c400="#9ca3af", c50="#ecf1e8", c500="#1E2D2F", c600="#1E2D2F", c700="#374151", c800="#1f2937", c900="#111827", c950="#0b0f19"),
)

# Gradioインターフェースの定義
iface = gr.Interface(
    fn=transcribe,
    # fn=None,
    inputs=gr.Audio(type="filepath", label="Upload Audio (MP3 or MP4)"),
    outputs="text",
    title="KotobaTranscriber",
    description=description,
    theme=theme,
)
# アプリの起動
iface.launch(server_name="0.0.0.0", server_port=7860, share=True)