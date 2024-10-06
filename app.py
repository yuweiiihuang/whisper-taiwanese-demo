import gradio as gr

# 環境設定
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"     # 啟用 Apple Metal (MPS) 的後備模式

import tempfile
import torch
from peft import PeftModel, PeftConfig
from transformers import (
    pipeline,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
    WhisperProcessor,
)
from yt_dlp import YoutubeDL

# 確認支援的運算設備
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 參數設定
yt_length_limit_s = 600  # 限制 YouTube 影片的最大長度為 10 分鐘
peft_model_id = "yuweiiizz/whisper-small-taiwanese-lora"
language = "Chinese"
task = "transcribe"

# 模型與處理器初始化
peft_config = PeftConfig.from_pretrained(peft_model_id)
model = WhisperForConditionalGeneration.from_pretrained(peft_config.base_model_name_or_path, device_map=device)
model = PeftModel.from_pretrained(model, peft_model_id)
tokenizer = WhisperTokenizer.from_pretrained(peft_config.base_model_name_or_path, language=language, task=task)
processor = WhisperProcessor.from_pretrained(peft_config.base_model_name_or_path, language=language, task=task)
feature_extractor = processor.feature_extractor

pipe = pipeline(
    "automatic-speech-recognition",
    model=model, 
    tokenizer=tokenizer, 
    feature_extractor=feature_extractor,
    max_new_tokens=128,
    chunk_length_s=15
    )

# 轉錄音訊檔案的功能
def transcribe(microphone=None, file_upload=None):
    if microphone and file_upload:
        warn_output = "警告：您同時使用了麥克風與上傳音訊檔案，將只會使用麥克風錄製的檔案。\n"
        file = microphone
    elif microphone or file_upload:
        warn_output = ""
        file = microphone if microphone else file_upload
    else:
        return "錯誤：您必須至少使用麥克風或上傳一個音訊檔案。"

    text = pipe(file, generate_kwargs={"task": task, "language": language})["text"]
    return warn_output + text

# 轉錄 YouTube 影片的功能
def yt_transcribe(yt_url):
    try:
        import stable_whisper

        # 使用 yt-dlp 下載音訊
        ydl_opts = {
            'format': 'bestaudio/best',
            'noplaylist': True,
            'quiet': True,
            'outtmpl': os.path.join(tempfile.gettempdir(), '%(id)s.%(ext)s'),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
        }
        with YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(yt_url, download=True)
            audio_path = ydl.prepare_filename(info_dict)
            audio_path = os.path.splitext(audio_path)[0] + ".mp3"

        # 定義轉錄函數
        def inference(audio, **kwargs) -> dict:
            pipe_output = pipe(audio, generate_kwargs={"task": task, "language": language}, return_timestamps=True)
            chunks = [{"start": c['timestamp'][0] or 0, "end": c['timestamp'][1] or c['timestamp'][0] + 5, "text": c['text']} for c in pipe_output['chunks']]
            return chunks

        # 使用 stable_whisper 進行轉錄
        result = stable_whisper.transcribe_any(inference, audio_path, vad=True)
        os.remove(audio_path)
        
        # 解析 URL 中的 video ID
        video_id = info_dict.get('id', None)
        if not video_id:
            return "錯誤：無法解析 YouTube 影片 ID。", "", None
        
        # 嵌入 YouTube 影片
        html_embed = f'<center><iframe width="500" height="320" src="https://www.youtube.com/embed/{video_id}"></iframe></center>'
        
        # 格式化字幕並保存為 SRT 檔案
        srt_text = result.to_srt_vtt(word_level=False)
        srt_path = os.path.join(tempfile.gettempdir(), f"{video_id}.srt")
        with open(srt_path, 'w') as srt_file:
            srt_file.write(srt_text)
        
        return html_embed, srt_text, srt_path
    except Exception as e:
        return f"錯誤：處理 YouTube 影片時發生錯誤。錯誤詳情：{str(e)}", "", None

# 範例音訊檔案
example_paths = [
    ["examples/common_voice_1.mp3"],
    ["examples/common_voice_2.mp3"],
    ["examples/common_voice_3.mp3"],
    ["examples/dictionary_1.mp3"],
    ["examples/dictionary_2.mp3"],
    ["examples/dictionary_3.mp3"],
]

example_info = """
| Example File | 台語漢字 | 華語 | 拼音 |
|--------------|------------|------|------|
| common_voice_1.mp3 | 我欲學臺語 | 我要學臺語 | guá beh o̍h Tâi-gí |
| common_voice_2.mp3 | 有這款的代誌,我攏毋知 | 有這種事情,我都不知道 | ū tsit-khuán ê tāi-tsì, guá lóng m̄ tsai |
| common_voice_3.mp3 | 豐原 | 豐原 | Hong-guân |
| dictionary_1.mp3 | 你今仔日下晡去佗愛交代清楚。 | 你今天下午到哪去要說明清楚。 | Lí kin-á-ji̍t ē-poo khì toh ài kau-tài tshing-tshó. |
| dictionary_2.mp3 | 𠢕眩船的人愛食眩船藥仔。 | 容易暈船的人要吃暈船藥。 | Gâu hîn-tsûn ê lâng ài tsia̍h hîn-tsûn io̍h-á |
| dictionary_3.mp3 | 三分天註定，七分靠拍拚。 | 三分天注定，七分靠努力。 | Sann hun thinn tsù-tiānn, tshit hun khò phah-piànn. |
"""

# 獲取範例檔案的絕對路徑
script_dir = os.path.dirname(os.path.abspath(__file__))
examples = [os.path.join(script_dir, example_path[0]) for example_path in example_paths]

# Gradio 介面
demo = gr.Blocks()

mf_transcribe = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(label="audio", type="filepath"),
    outputs="text",
    title="Whisper 台語演示: 語音轉錄",
    description=f"演示使用 `PEFT-LoRA` fine-tuned checkpoint [{peft_model_id}](https://huggingface.co/{peft_model_id} 轉錄任意長度的音訊檔案",
    allow_flagging="manual",
    examples=examples,
    article=example_info,
)

yt_transcribe = gr.Interface(
    fn=yt_transcribe,
    inputs=[gr.Textbox(lines=1, placeholder="在此處貼上 YouTube 影片的 URL", label="YouTube URL")],
    outputs=[
        gr.HTML(label="YouTube Video Embed"),
        gr.Textbox(label="轉錄稿"),
        gr.File(label="下載 SRT 檔案")
    ],
    title="Whisper 台語演示: Youtube轉錄",
    description=f"演示使用 `PEFT-LoRA` fine-tuned checkpoint [{peft_model_id}](https://huggingface.co/{peft_model_id} 轉錄任意長度的Youtube影片",
    allow_flagging="manual",
)

with demo:
    gr.TabbedInterface([mf_transcribe, yt_transcribe], ["語音轉錄", "Youtube轉錄"])

demo.launch(share=True)
