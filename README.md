# Vietnamese TTS Studio

Flask webapp cho text-to-speech tiếng Việt, dùng hai engine:

- `F5-TTS` cho zero-shot voice cloning linh hoạt.
- `ViRa` cho tiếng Việt tự nhiên hơn.

## Run

```powershell
python -m pip install -r requirements.txt
python webapp/app.py
```

App chạy tại `http://127.0.0.1:8386`.

## Engine setup

### F5-TTS

```powershell
python -m pip install f5-tts
```

Biến môi trường hỗ trợ:

- `F5_MODEL_NAME`
- `F5_CKPT_FILE`
- `F5_VOCAB_FILE`
- `F5_VOCODER_LOCAL_PATH`

### ViRa

```powershell
python -m pip install git+https://github.com/iamdinhthuan/Vira-tts.git
```

Nếu dùng local model:

- đặt model vào `models/vira`
- hoặc cấu hình `VIRA_MODEL_PATH`

Nếu muốn app tự tải model:

```powershell
$env:VIRA_AUTO_DOWNLOAD = "1"
```

## API

- `GET /api/tts/status`
- `POST /api/tts/generate`
- `GET /outputs/<filename>`

## Runtime folders

- `webapp/runtime/references`: file audio tham chiếu upload lên
- `webapp/runtime/generated`: audio WAV đầu ra
