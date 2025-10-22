from src.data.clean_data import CleanDataService
from pathlib import Path

def test_clean_data():
    cleaner = CleanDataService(Path("configs/config.yaml"))
    cleaner.run()
    assert cleaner.dest_path.exists()
