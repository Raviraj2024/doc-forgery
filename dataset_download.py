import kagglehub

print("Starting download via kagglehub...")
# Download latest version, inherently handles streaming and caching
path = kagglehub.dataset_download("dinmkeljiame/doctamper")

print("Path to dataset files:", path)