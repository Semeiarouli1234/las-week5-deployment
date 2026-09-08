# Lokasi file model

## Tomato Leaf (MobileNetV2 Fine-Tuned)
Taruh di: `models/tomato_mobilenetv2_ft/model.keras`

Kalau hasil export dari Kaggle formatnya beda (misal `.h5`), ganti nama file
sesuai, lalu update `MODEL_PATH` di `utils/tomato_model.py`.

## Sentimen MBG (IndoBERT Fine-tuned)
Taruh SELURUH folder hasil `trainer.save_model()` + `tokenizer.save_pretrained()`
di: `models/indobert_sentiment/`

Folder ini biasanya berisi: `config.json`, `model.safetensors` (atau
`pytorch_model.bin`), `tokenizer_config.json`, `vocab.txt`, dll.
