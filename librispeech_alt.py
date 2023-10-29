import csv
import re
import os
from pathlib import Path
from typing import Tuple, Union

from torch import Tensor
from torch.hub import download_url_to_file
from torch.utils.data import Dataset
from torchaudio.datasets.utils import _extract_tar, _load_waveform

URL = "train-clean-100"
FOLDER_IN_ARCHIVE = "cv-corpus-15.0-2023-09-08"
SAMPLE_RATE = 32000  # 48000
_DATA_SUBSETS = [
    "train",
    "dev",
    "test",
    "validated",
    "other"
]


def _get_librispeech_metadata(
    fileid: str, root: str, folder: str, ext_audio: str, ext_txt: str
) -> Tuple[str, int, str, int, int, int]:
    speaker_id, chapter_id, utterance_id = fileid.split("-")

    # Get audio path and sample rate
    fileid_audio = f"{speaker_id}-{chapter_id}-{utterance_id}"
    filepath = os.path.join(folder, speaker_id, chapter_id, f"{fileid_audio}{ext_audio}")

    # Load text
    file_text = f"{speaker_id}-{chapter_id}{ext_txt}"
    file_text = os.path.join(root, folder, speaker_id, chapter_id, file_text)
    with open(file_text) as ft:
        for line in ft:
            fileid_text, transcript = line.strip().split(" ", 1)
            if fileid_audio == fileid_text:
                break
        else:
            # Translation not found
            raise FileNotFoundError(f"Translation not found for {fileid_audio}")

    return (
        filepath,
        SAMPLE_RATE,
        transcript,
        int(speaker_id),
        int(chapter_id),
        int(utterance_id),
    )


class LIBRISPEECH(Dataset):
    """*LibriSpeech* :cite:`7178964` dataset.

    Args:
        root (str or Path): Path to the directory where the dataset is found or downloaded.
        url (str, optional): The URL to download the dataset from,
            or the type of the dataset to dowload.
            Allowed type values are ``"dev-clean"``, ``"dev-other"``, ``"test-clean"``,
            ``"test-other"``, ``"train-clean-100"``, ``"train-clean-360"`` and
            ``"train-other-500"``. (default: ``"train-clean-100"``)
        folder_in_archive (str, optional):
            The top-level directory of the dataset. (default: ``"LibriSpeech"``)
        download (bool, optional):
            Whether to download the dataset if it is not found at root path. (default: ``False``).
    """

    _ext_txt = ".trans.txt"
    _ext_audio = ".mp3"

    def __init__(
        self,
        root: Union[str, Path],
        language: str,
        split: str = "train",
        folder_in_archive: str = FOLDER_IN_ARCHIVE,
        download: bool = False,
    ) -> None:
        self._split = split
        if split not in _DATA_SUBSETS:
            raise ValueError(f"Invalid url '{split}' given; please provide one of {_DATA_SUBSETS}.")

        root = os.fspath(root)
        self._archive = os.path.join(root, folder_in_archive, language, "clips")
        self._path = os.path.join(root, folder_in_archive, language, f"{split}.tsv")

        self._language = language

        if not os.path.isfile(self._path):
            if download:
                pass
            else:
                raise RuntimeError(
                    f"Dataset not found at {self._path}. Please set `download=True` to download the dataset."
                )

        with open(self._path, "r") as f:
            reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            self._data = list(reader)

    def get_metadata(self, n: int) -> Tuple[Tensor, int, str, int, int, int]:
        """Get metadata for the n-th sample from the dataset. Returns filepath instead of waveform,
        but otherwise returns the same fields as :py:func:`__getitem__`.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            Tuple of the following items;

            str:
                Path to audio
            int:
                Sample rate
            str:
                Transcript
            int:
                Speaker ID
            int:
                Chapter ID
            int:
                Utterance ID
        """
        fileid = self._walker[n]
        return _get_librispeech_metadata(fileid, self._archive, self._url, self._ext_audio, self._ext_txt)

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, int, int, int]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            Tuple of the following items;

            Tensor:
                Waveform
            int:
                Sample rate
            str:
                Transcript
            int:
                id
        """
        row = self._data[n]
        id_match = re.match(f"common_voice_{self._language}_(\d+).mp3", row["path"])
        if id_match:
            id = id_match.group(1)
        waveform = _load_waveform(self._archive, row["path"], SAMPLE_RATE)
        return (waveform, SAMPLE_RATE, row["sentence"], id)

    def __len__(self) -> int:
        return len(self._data)