import os
import cv2
import numpy as np
from tqdm import tqdm
import pickle

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchaudio

from . import DATASETS
from .utils import DataCollection, af_pad_sequence


@DATASETS.register("RealLife")
class RealLife_Dataset(Dataset):
    def __init__(
        self,
        dataset_path,
        transform,
        frame_size=160,
        n_sample_frames=64,
        modalities=["visual", "audio"],
    ) -> None:
        """_summary_

        Args:
            path (str): path to reallife dataset
        """
        super(RealLife_Dataset, self).__init__()
        self.path = dataset_path
        self.frame_size = frame_size
        self.n_sample_frames = n_sample_frames
        self.modalities = modalities

        self.tags = ["Deceptive", "Truthful"]
        self.clip_files = []
        self.audio_files = []
        self.labels = []
        for tag in sorted(self.tags):
            _files = os.listdir(os.path.join(self.path, "Clips", tag))
            _files = sorted(
                [os.path.join(self.path, "Clips", tag, file) for file in _files]
            )
            self.clip_files += _files

            if tag == "Deceptive":
                label = 1
            else:
                label = 0
            self.labels += [label] * len(_files)

            _files = [
                file.replace("Clips", "Audios")[:-4] + "_audio.wav" for file in _files
            ]
            self.audio_files += _files
        assert len(self.labels) == len(self.clip_files) == len(self.audio_files)

        self.transform = transform
        self.is_memory_sufficient = False

    def __len__(self):
        return len(self.labels)

    def memory_sufficient(self, args):
        self.is_memory_sufficient = True
        if args.model in ["ViT_model", "PECL"]:
            suffix = "_ViT"
        else:
            suffix = "_VideoMAE"
        if not os.path.exists(os.path.join("cache", f"RealLife{suffix}.pkl")):
            self.clips = []
            for clip_file in tqdm(self.clip_files):
                frames = self.read_vid(clip_file)
                self.clips.append(frames)
            self.clips = torch.stack(self.clips)
            if not os.path.exists("cache"):
                os.makedirs("cache")
            with open(os.path.join("cache", f"RealLife{suffix}.pkl"), "wb") as f:
                pickle.dump(self.clips, f)
        else:
            with open(os.path.join("cache", f"RealLife{suffix}.pkl"), "rb") as f:
                self.clips = pickle.load(f)        
        if args.model in ["PECL"]:
            suffix = "_ViT"
        else:
            suffix = ""
        if not os.path.exists(os.path.join("cache", f"RealLife-audio{suffix}.pkl")):
            self.audios = []
            for audio_file in tqdm(self.audio_files):
                audio = self.read_aud(audio_file)
                self.audios.append(audio)
            self.audios = af_pad_sequence(self.audios)
            with open(os.path.join("cache", f"RealLife-audio{suffix}.pkl"), "wb") as f:
                pickle.dump(self.audios, f)
        else:
            with open(os.path.join("cache", f"RealLife-audio{suffix}.pkl"), "rb") as f:
                self.audios = pickle.load(f)

    def read_aud(self, path):
        waveform, sample_rate = torchaudio.load(path)
        waveform = waveform[0]
        clip_duration = len(waveform) / sample_rate
        new_sample_rate = int(
            np.round(
                321.893491124260 * self.n_sample_frames / clip_duration, decimals=0
            )
        )
        waveform = torchaudio.functional.resample(
            waveform, sample_rate, new_sample_rate
        )
        mono_waveform = waveform.unsqueeze(0)
        mono_waveform.type(torch.float32)
        return mono_waveform

    def read_vid(self, path):
        vid = cv2.VideoCapture(path)
        frames = []
        while vid.isOpened():
            ret, frame = vid.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(self.transform(frame))
            else:
                break
        target_frames = np.linspace(0, len(frames) - 1, num=self.n_sample_frames)
        target_frames = np.around(target_frames).astype(int)
        frames = [frames[idx] for idx in target_frames]
        # frames = self.transform(frames, return_tensors="pt")["pixel_values"].squeeze(0)
        frames = torch.stack(frames, 0)
        frames.type(torch.float32)
        return frames

    def __getitem__(self, index):
        if "audio" in self.modalities:
            if not self.is_memory_sufficient:
                audio = self.read_aud(self.audio_files[index])
            else:
                audio = self.audios[index]
        else:
            audio = None
        if "visual" in self.modalities:
            if not self.is_memory_sufficient:
                video = self.read_vid(self.clip_files[index])
            else:
                video = self.clips[index]
        else:
            video = None
        return DataCollection(visual=video, audio=audio, label=self.labels[index])
