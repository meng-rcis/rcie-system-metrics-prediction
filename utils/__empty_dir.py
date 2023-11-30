import os


def delete_empty_folders(start_path):
    """
    Delete all empty folders recursively starting from the given path.

    Parameters:
    - start_path: The path from where to start checking for empty folders.
    """
    for dirpath, dirnames, filenames in os.walk(start_path, topdown=False):
        # topdown=False makes os.walk to traverse the directories from the deepest to the shallowest
        if not dirnames and not filenames:
            print(f"Deleting empty directory: {dirpath}")
            os.rmdir(dirpath)
        else:
            print(f"Not empty, skipping: {dirpath}")


# Run at the root directory of the project - cmd: python .\utils\empty_dir.py
delete_empty_folders("./models/features/source")
