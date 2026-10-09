import kagglehub

# Download latest version
path = kagglehub.dataset_download("amellia/face-skin-disease")

print("Path to dataset files:", path)