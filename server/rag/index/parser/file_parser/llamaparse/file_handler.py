import shutil
from abc import ABC, abstractmethod


class FileHandler(ABC):
    @abstractmethod
    def download_file(self, file_path: str, destination_path: str) -> None:
        pass

    @abstractmethod
    def upload_file(self, file_path: str, destination_path: str) -> None:
        pass

    @abstractmethod
    def sync_foler(self, source: str, destination: str) -> None:
        pass


class LocalHandler(FileHandler):
    def download_file(self, file_path: str, destination_path: str) -> None:
        shutil.copy(file_path, destination_path)

    def upload_file(self, file_path: str, destination_path: str) -> None:
        shutil.copy(file_path, destination_path)

    def sync_foler(self, source: str, destination: str) -> None:
        shutil.copytree(source, destination, dirs_exist_ok=True)
