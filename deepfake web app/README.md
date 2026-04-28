# DeepFake Web App

Simple Flask app for your trained `deepfake_cnn_lstm.pth` model.

## Folders

- `input/` stores uploaded user files
- `outputs/` stores JSON prediction outputs

## Run

```powershell
cd "e:\Verifixia\deepfake web app"
.\run_app.bat
```

Then open `http://127.0.0.1:5000`.

## Supported files

- Images: `png`, `jpg`, `jpeg`, `gif`, `bmp`, `webp`
- Videos: `mp4`, `avi`, `mov`, `mkv`, `webm`

## Output

Each uploaded file is saved under `input/`.

Each prediction is saved as a JSON file under `outputs/`.
