# Vietnamese TTS Studio

Flask webapp cho text-to-speech tiếng Việt, dùng hai engine:

- `VieNeu-TTS` làm engine mặc định, hiện ưu tiên mode `standard` cho chất lượng.
- `F5-TTS` giữ lại như engine phụ để thử model hoặc flow khác.

## Run

```powershell
python -m pip install -r requirements.txt
python webapp/app.py
```

App chạy tại `http://127.0.0.1:8386`.

## Colab notebook

Notebook Colab để clone hoặc pull repo, chạy webapp và mở Cloudflare tunnel:

- `colab_tts_cloudflare.ipynb`

Nếu gặp lỗi kiểu `Codec 'neuphonic/distill-neucodec' requires PyTorch`, `torch >= 2.11.0`, hoặc `libtorchaudio.so: undefined symbol: torch_list_push_back`, hãy dùng notebook mới nhất:

- notebook đã chuyển sang stack `torch>=2.11.0`, `torchaudio>=2.11.0` và index CUDA 12.8 (`cu128`) theo hướng upstream của VieNeu
- notebook cũng uninstall lại bộ `torch/neucodec/vieneu` cũ trước khi cài để tránh lệch binary ABI
- notebook hiện mặc định `TTS_DEFAULT_ENGINE=vieneu`, `VIENEU_MODE=standard`, và không cài F5 nếu bạn không bật lại thủ công

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
- `TTS_EXPOSE_F5=1` nếu muốn vẫn hiện card F5 trên UI dù engine này chưa sẵn sàng

### VieNeu-TTS

```powershell
python -m pip install -U "vieneu[gpu]==2.4.3"
```

Biến môi trường hỗ trợ:

- `VIENEU_MODE` mặc định là `standard`
- `VIENEU_MODE_CHOICES` thêm preset mode trên UI, ví dụ `Standard=standard;Fast=fast;Turbo=turbo;Turbo GPU=turbo_gpu`

Khuyến nghị:

- giữ `VIENEU_MODE=standard` nếu ưu tiên chất lượng và clone giọng sát hơn
- chỉ đổi về `turbo` nếu runtime không đủ dependency cho Standard hoặc không có GPU
- với Colab, rerun lại cell install nếu trước đó runtime từng cài torch 2.6/cu124 hoặc notebook cũ
- dùng audio tham chiếu dài khoảng 3-8 giây, một người nói, ít nhạc nền
- transcript tham chiếu là bắt buộc với `standard` và `fast`

## API

- `GET /api/tts/status`
- `POST /api/tts/generate` nhận thêm `model_key` và `custom_model` để chọn model theo từng request
- `GET /outputs/<filename>`

## Runtime folders

- `webapp/runtime/references`: file audio tham chiếu upload lên
- `webapp/runtime/generated`: audio WAV đầu ra
