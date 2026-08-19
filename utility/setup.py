import os
# Define project files relative to root
# create a basic list of files to be created in the project structure.
files = [
    "README.md",
    ".gitignore",
    ".github/workflows/ci.yml",
    "src/train.py",
    "src/model.py",
    "src/dataset.py",
    "src/serve.py",
    "configs/training_config.yaml",
    "docker/Dockerfile.train",
    "docker/Dockerfile.serve",
    "k8s/namespace.yaml",
    "k8s/training-job.yaml",
    "k8s/serving-deployment.yaml",
    "k8s/serving-service.yaml",
    "k8s/configmap.yaml",
    "k8s/hpa.yaml",
    "requirements/train.txt",
    "requirements/serve.txt",
    "tests/test_model.py",
    "utility/setup.py",
    "utility/logging.py"
]

# Create directories and empty files
for file_path in files:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "a"):
        os.utime(file_path, None)  # Creates an empty file if it doesn't exist

print("Project structure created successfully!")