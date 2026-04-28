import os

path = r"e:\Verifixia\Frontend\src\pages\Dashboard.tsx"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# Fix threatLevel reset bug
content = content.replace(
"""  // Real-time frame capture for live monitoring
  useEffect(() => {
    if (!isMonitoring || !!mediaSrc) {
      setThreatLevel("safe");
      return;
    }""",
"""  // Real-time frame capture for live monitoring
  useEffect(() => {
    if (!isMonitoring || !!mediaSrc) {
      if (!mediaSrc && !isMonitoring) {
        setThreatLevel("safe");
      }
      return;
    }"""
)

# Fix URL for image preview to use local Object URL always
old_handle_upload = """      if (result?.file_url) {
        clearCurrentObjectUrl();
        setMediaSrc(result.file_url);
        setMediaType(result.isVideo ? "video" : "image");
      } else {
        const localUrl = URL.createObjectURL(file);
        setCurrentObjectUrl(localUrl);
        setMediaSrc(localUrl);
        setMediaType(result?.isVideo ? "video" : "image");
      }"""

new_handle_upload = """      const localUrl = URL.createObjectURL(file);
      clearCurrentObjectUrl();
      setCurrentObjectUrl(localUrl);
      setMediaSrc(localUrl);
      setMediaType(result?.isVideo ? "video" : "image");"""

content = content.replace(old_handle_upload, new_handle_upload)

# Add processing indicator (fluctuating score and previewing immediately)
old_try_block = """    try {
      const result = await uploadImage(file);"""

new_try_block = """    const localUrlTemp = URL.createObjectURL(file);
    setCurrentObjectUrl(localUrlTemp);
    setMediaSrc(localUrlTemp);
    setMediaType(file.type.startsWith("video/") ? "video" : "image");

    let isProcessing = true;
    const fluctuateInterval = setInterval(() => {
      if (isProcessing) {
        setConfidenceScore(10 + Math.floor(Math.random() * 80));
      }
    }, 150);

    try {
      const result = await uploadImage(file);
      isProcessing = false;
      clearInterval(fluctuateInterval);"""

content = content.replace(old_try_block, new_try_block)

old_catch_block = """    } catch (error) {
      console.error("Upload failed", error);
      setThreatLevel("safe");"""

new_catch_block = """    } catch (error) {
      isProcessing = false;
      clearInterval(fluctuateInterval);
      console.error("Upload failed", error);
      setThreatLevel("safe");
      setConfidenceScore(0);"""

content = content.replace(old_catch_block, new_catch_block)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)

print("Patch applied successfully.")
