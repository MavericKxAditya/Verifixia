import os

path = r"e:\Verifixia\Backend\app.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

old_block = """    # Save frames to temp files and run image prediction on each
    import tempfile
    fake_scores = []
    for i, frame_img in enumerate(frames_extracted):
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                frame_img.save(tmp.name, "JPEG")
                tmp_path = tmp.name
            frame_result = predict_deepfake(tmp_path)
            os.remove(tmp_path)
            fake_scores.append(frame_result.get("confidence_raw", 0.5))
        except Exception as e:
            logger.warning(f"Frame {i} prediction failed: {e}")

    if not fake_scores:
        return "Unknown", 0.5

    avg_score = sum(fake_scores) / len(fake_scores)
    prediction = "Fake" if avg_score > 0.55 else "Real"
    return prediction, avg_score"""

new_block = """    # Save frames to temp files and run image prediction on each
    import tempfile
    predictions = []
    confidences = []
    for i, frame_img in enumerate(frames_extracted):
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                frame_img.save(tmp.name, "JPEG")
                tmp_path = tmp.name
            frame_result = predict_deepfake(tmp_path)
            os.remove(tmp_path)
            predictions.append(frame_result.get("prediction", "Unknown"))
            confidences.append(frame_result.get("confidence", 50.0))
        except Exception as e:
            logger.warning(f"Frame {i} prediction failed: {e}")

    if not predictions:
        return "Unknown", 0.5

    fake_count = sum(1 for p in predictions if p == "Fake")
    prediction = "Fake" if fake_count > len(predictions) / 2 else "Real"
    
    # predict_deepfake returns confidence as 0-100 percentage.
    # We must return a 0-1 fraction because upload_image multiplies it by 100.
    avg_confidence = sum(confidences) / len(confidences) / 100.0
    return prediction, avg_confidence"""

content = content.replace(old_block, new_block)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)

print("Backend patch applied successfully.")
