# Vietnamese TTS Studio

Flask webapp cho text-to-speech tiếng Việt, hiện hỗ trợ ba engine:

- `Gwen-TTS` là model chủ chốt, ưu tiên voice cloning tiếng Việt với transcript tham chiếu.
- `VieNeu-TTS` giữ vai trò engine phụ cho flow tiếng Việt 24kHz.
- `F5-TTS` vẫn có sẵn để thử model hoặc flow khác.

## Run

```powershell
python -m pip install -r requirements.txt
python webapp/app.py
```

App chạy tại `http://127.0.0.1:8386`.

## Colab notebook

Notebook Colab để clone hoặc pull repo, chạy webapp và mở Cloudflare tunnel:

- `colab_tts_cloudflare.ipynb`

Nếu gặp lỗi kiểu `Codec 'neuphonic/distill-neucodec' requires PyTorch`, `torch >= 2.11.0`, `libtorchaudio.so: undefined symbol: torch_list_push_back`, hoặc thiếu stack cho `qwen-tts`, hãy dùng notebook mới nhất:

- notebook hiện mặc định `TTS_DEFAULT_ENGINE=gwen`, chỉ cài `qwen-tts`, và export `TTS_ENABLED_ENGINES=gwen` để runtime Colab chỉ hiện engine chủ đạo
- notebook pin đúng cặp `torch==2.11.0`, `torchaudio==2.11.0` và index CUDA 12.8 (`cu128`) theo hướng upstream của VieNeu
- notebook cũng dọn residue `torch/torchaudio/neucodec/vieneu` cũ trước khi cài và verify stack trong subprocess sạch để tránh lệch binary ABI
- notebook export thêm `GWEN_MODEL_ID` và `GWEN_ATTN_IMPLEMENTATION` để backend load Gwen-TTS theo đúng runtime hiện tại

## Engine setup

### Gwen-TTS

```powershell
python -m pip install -U qwen-tts
```

Gwen-TTS là model mặc định của project. Engine này cần:

- GPU CUDA để chạy local
- audio tham chiếu khoảng 3-10 giây, một người nói, ít nhạc nền
- transcript tham chiếu đúng với câu đang có trong audio mẫu

Web hiện có sẵn 9 preset voice chính thức của Gwen-TTS, lấy từ demo/repo gốc, và thêm 1 preset custom `Khả Hân`:

- `Yến Nhi`
- `Mỹ Vân`
- `Ái Vy`
- `An Nhi`
- `Diệu Linh`
- `Khánh Toàn`
- `Trần Lâm`
- `NSND Hà Phương`
- `NSND Kim Cúc`
- `Khả Hân`

Metadata nằm ở `webapp/data/gwen_preset_voices.json`, audio preview nằm ở `webapp/static/voice_presets`.

Biến môi trường hỗ trợ:

- `GWEN_MODEL_ID` mặc định là `g-group-ai-lab/gwen-tts-0.6B`
- `GWEN_MODEL_CHOICES` thêm preset model Gwen trên UI, ngăn cách bằng `;`, hỗ trợ `label=value`
- `GWEN_DTYPE` mặc định `bfloat16`
- `GWEN_ATTN_IMPLEMENTATION` mặc định `flash_attention_2`; app sẽ fallback sang `sdpa` nếu load không được

Nếu runtime không có GPU CUDA hoặc không cài được `flash-attn`, hãy giữ `GWEN_ATTN_IMPLEMENTATION=sdpa` hoặc đổi sang VieNeu/F5.

### F5-TTS

```powershell
python -m pip install f5-tts
```

Nếu gặp lỗi kiểu `No module named 'cached_path'`, môi trường đang có khả năng đã cài `f5-tts` thiếu dependency. Hãy cài lại:

```powershell
python -m pip install -U f5-tts cached_path
```

Trong Colab notebook của repo này, không nên cài `f5-tts` với `--no-deps`.

App hiện sẽ tự chuẩn hóa reference audio của F5 sang WAV mono 24kHz trước khi infer. Nếu bạn upload `mp3/m4a/ogg`, runtime vẫn cần có `ffmpeg`; còn nếu không có `ffmpeg`, hãy dùng `wav/flac`.

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

Nếu bạn cần chạy `standard/fast` trên Colab, hãy dùng cell install mới của notebook. `pip install vieneu[gpu]` thường fail vì pip không tự resolve các wheel GPU mà upstream map qua `uv` sources.

Biến môi trường hỗ trợ:

- `VIENEU_MODE` mặc định là `standard`
- `VIENEU_MODE_CHOICES` thêm preset mode trên UI, ví dụ `Standard=standard;Fast=fast;Turbo=turbo;Turbo GPU=turbo_gpu`

Khuyến nghị:

- giữ `VIENEU_MODE=standard` nếu cần engine phụ chất lượng cao và chấp nhận nhập transcript
- chỉ đổi về `turbo` nếu runtime không đủ dependency cho Standard hoặc không có GPU
- với Colab, rerun lại cell install nếu trước đó runtime từng cài torch 2.6/cu124 hoặc notebook cũ
- dùng audio tham chiếu dài khoảng 3-8 giây, một người nói, ít nhạc nền
- transcript tham chiếu là bắt buộc với `standard` và `fast`

## API

- `GET /api/tts/status`
- `POST /api/tts/generate` nhận thêm `model_key` và `custom_model` để chọn model theo từng request
- `GET /outputs/<filename>`

## UI routes

- `/studio/gwen`
- `/studio/vieneu`
- `/studio/f5`

## Runtime folders

- `webapp/runtime/references`: file audio tham chiếu upload lên
- `webapp/runtime/generated`: audio WAV đầu ra
