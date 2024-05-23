from pathlib import Path

def create_directory(directory_path):
    # Create a Path object
    path = Path(directory_path)
    
    # Check if the directory exists
    if not path.exists():
        # Create the directory
        path.mkdir(parents=True, exist_ok=True)
        print(f"Directory '{directory_path}' created successfully.")
    else:
        print(f"Directory '{directory_path}' already exists.")

