# DeepFakeTester

Drop custom media into `input/` and run the tester against the trained model at `../Notebooks/deepfake_cnn_lstm.pth`.

## Supported input

- Images: `png`, `jpg`, `jpeg`, `gif`, `bmp`, `webp`
- Videos: `mp4`, `avi`, `mov`, `mkv`, `webm`

## Run

```bash
python tester.py
```

If `python` points to a different environment on your machine, use the bundled launchers instead:

```powershell
.\run_tester.bat
```

or:

```powershell
.\run_tester.ps1
```

## Output

- Per-file JSON result: `output/<filename>_result.json`
- Run summary: `output/summary_<timestamp>.json`

## Notes

- The model is loaded from `e:/Verifixia/Notebooks/deepfake_cnn_lstm.pth`
- Video inference samples `8` evenly spaced frames per file
- Preprocessing uses `224x224` resize and ImageNet normalization
- The tested working environment here is Python `3.12` with `torch 2.11.0` and `torchvision 0.26.0`
