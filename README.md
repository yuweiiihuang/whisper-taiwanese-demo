# Whisper 台語轉錄演示

這是一個基於 Gradio 的展示，使用經過 PEFT-LoRA 微調的 Whisper 模型來轉錄台語音檔檔案和 YouTube 影片。

## 目錄

- [功能](#功能)
- [前置需求](#前置需求)
- [安裝](#安裝)
  - [1. 複製存儲庫](#1-複製存儲庫)
  - [2. 設置虛擬環境](#2-設置虛擬環境)
  - [3. 安裝依賴套件](#3-安裝依賴套件)
- [使用方法](#使用方法)
  - [執行應用程式](#執行應用程式)
  - [使用網頁介面](#使用網頁介面)
- [專案結構](#專案結構)

## 功能

- **音檔轉錄**：將上傳或使用麥克風錄製的台語音檔轉換為華語。
- **YouTube 轉錄**：轉錄 YouTube 影片中的音檔並生成 SRT 字幕檔案。
- **語言支援**：專門針對台語進行了 PEFT-LoRA 微調。
- **裝置相容性**：支援 Apple Metal (MPS)、CUDA 及 CPU 進行運算。
- **使用者介面**：使用 Gradio 架設互動式的網頁介面。

## 前置需求

- **Python**：版本 3.11 或更高。
- **硬體**：
  - **Apple Silicon Mac**：使用 Apple Metal Performance Shaders (MPS)。
  - **NVIDIA GPU**：需要 CUDA 以加速 GPU 運算。
  - **CPU**：若無 GPU 可使用 CPU 進行運算。
- **Git**：用於複製存儲庫。

## 安裝

### 1. 複製存儲庫

首先，使用 Git 將存儲庫複製到本機：

```bash
git clone https://github.com/yuweiiihuang/whisper-taiwanese-demo.git
cd whisper-taiwanese-demo
```

### 2. 設置虛擬環境

建議使用虛擬環境來管理依賴套件。

#### 使用 `venv`

```bash
python3 -m venv venv
source venv/bin/activate  # Windows 使用者請執行: venv\Scripts\activate
```

#### 使用 `conda`

```bash
conda create -n whisper-env python=3.11
conda activate whisper-env
```

### 3. 安裝依賴套件

使用 `pip` 安裝所需的 Python 套件。確保已安裝最新版本的 `pip`：

```bash
pip install --upgrade pip
```

然後，安裝依賴套件：

```bash
pip install -r requirements.txt
```

**注意**：可能需要安裝特定版本的 `torch` 以支援不同的裝置（如: CUDA）。

更多詳情請參閱 [PyTorch 安裝指南](https://pytorch.org/get-started/locally/)。

## 使用方法

### 執行應用程式

安裝完依賴套件後，可以執行應用程式：

```bash
python app.py
```

成功執行後，Gradio 會啟動 localhost 並提供一個 URL（例如 `http://127.0.0.1:7860`），可以在該網址訪問網頁介面。

使用 Gradio 的 `share=True` 參數可以生成一個臨時的公開連結，使介面可以透過網際網路存取。

### 使用網頁介面

網頁介面包含兩個主要功能：

1. **語音轉錄**：
   - **輸入**：上傳台語音檔或使用麥克風錄製。
   - **輸出**：轉錄後的華語。
   - **範例**：預載的音檔檔案展示轉錄範例。

2. **YouTube 轉錄**：
   - **輸入**：輸入 YouTube 影片的 URL，**必須符合以下格式**：

     ``` plaintext
     https://www.youtube.com/watch?v={video_id}
     ```

     如：

     ``` plaintext
     https://www.youtube.com/watch?v=1-9c7nMZZvM
     ```

   - **輸出**：
     - 嵌入的 YouTube 影片供參考。
     - 轉錄後的影片轉錄稿。
     - 可下載的 SRT 字幕檔案。

## 專案結構

``` plaintext
whisper-taiwanese-demo/
├── examples/
│   ├── common_voice_1.mp3
│   ├── common_voice_2.mp3
│   ├── common_voice_3.mp3
│   ├── dictionary_1.mp3
│   ├── dictionary_2.mp3
│   └── dictionary_3.mp3
├── app.py
├── README.md
└── requirements.txt
```

- **examples/**：包含用於測試轉錄的範例音檔。
- **app.py**：主要的 Python 腳本，包含 Gradio 介面和轉錄邏輯。
- **README.md**：專案文件。
- **requirements.txt**：列出所有 Python 依賴套件。
