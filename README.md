# Vietnamese TTS Studio

Flask webapp cho text-to-speech tiếng Việt, dùng hai engine:

- `F5-TTS` cho zero-shot voice cloning linh hoạt.
- `VieNeu-TTS` cho tiếng Việt ổn định hơn, ít lỗi runtime hơn ViRa.

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

Nếu gặp lỗi kiểu `No module named 'cached_path'`, môi trường đang có khả năng đã cài `f5-tts` thiếu dependency. Hãy cài lại:

```powershell
python -m pip install -U f5-tts cached_path
```

Trong Colab notebook của repo này, không nên cài `f5-tts` với `--no-deps`.

Biến môi trường hỗ trợ:

- `F5_MODEL_NAME`
- `F5_MODEL_CHOICES` (danh sách model hiển thị thêm trên UI, ngăn cách bằng `;`, hỗ trợ `label=value`)
- `F5_CKPT_FILE`
- `F5_VOCAB_FILE`
- `F5_VOCODER_LOCAL_PATH`

### VieNeu-TTS

```powershell
python -m pip install -U vieneu==2.4.3
```

Biến môi trường hỗ trợ:

- `VIENEU_MODE` mặc định là `turbo`
- `VIENEU_MODE_CHOICES` thêm preset mode trên UI, ví dụ `Turbo GPU=turbo_gpu;Standard=standard`

Khuyến nghị:

- giữ `VIENEU_MODE=turbo` nếu ưu tiên ổn định và ít lỗi
- dùng audio tham chiếu dài khoảng 3-8 giây, một người nói, ít nhạc nền
- transcript tham chiếu hiện không bắt buộc với `turbo`

## API

- `GET /api/tts/status`
- `POST /api/tts/generate` nhận thêm `model_key` và `custom_model` để chọn model theo từng request
- `GET /outputs/<filename>`

## Runtime folders

- `webapp/runtime/references`: file audio tham chiếu upload lên
- `webapp/runtime/generated`: audio WAV đầu ra
