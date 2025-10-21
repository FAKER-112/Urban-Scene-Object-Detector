import gdown
from pathlib import Path

def download_from_gdrive(url: str, output_path: str):
    """
    Downloads a file from a Google Drive shareable link.

    Args:
        url (str): Google Drive shareable URL.
        output_path (str): Local file path to save the download.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    try:
        gdown.download(url=url, output=output_path, quiet=False, fuzzy=True)
        print(f"Downloaded file to: {output_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to download from Google Drive: {e}")
