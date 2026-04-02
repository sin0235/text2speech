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

## Colab notebook

Notebook Colab để clone hoặc pull repo, chạy webapp và mở Cloudflare tunnel:

- `colab_tts_cloudflare.ipynb`

Nếu gặp lỗi `libtorchaudio.so: undefined symbol: torch_list_push_back`, hãy dùng notebook mới nhất:

- notebook đã pin `torch==2.6.0`, `torchvision==0.21.0`, `torchaudio==2.6.0`
- notebook cũng uninstall bộ `torch*` cũ trước khi cài lại để tránh lệch binary ABI

## Engine setup

### F5-TTS

```powershell
python -m pip install f5-tts
```

Biến môi trường hỗ trợ:

- `F5_MODEL_NAME`
- `F5_MODEL_CHOICES` (danh sách model hiển thị thêm trên UI, ngăn cách bằng `;`, hỗ trợ `label=value`)
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
- có thể thêm nhiều lựa chọn UI bằng `VIRA_MODEL_CHOICES`, ví dụ `Mặc định phụ=path:models/vira-alt;HF beta=repo:owner/repo`

Nếu muốn app tự tải model:

```powershell
$env:VIRA_AUTO_DOWNLOAD = "1"
```

## API

- `GET /api/tts/status`
- `POST /api/tts/generate` nhận thêm `model_key` và `custom_model` để chọn model theo từng request
- `GET /outputs/<filename>`

## Runtime folders

- `webapp/runtime/references`: file audio tham chiếu upload lên
- `webapp/runtime/generated`: audio WAV đầu ra
