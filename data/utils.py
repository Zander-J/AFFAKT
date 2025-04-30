import torch


class DataCollection:
    def __init__(self, visual, audio, label) -> None:
        self.visual = visual
        self.audio = audio
        self.label = label


def af_pad_sequence(batch):
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.0)
    return batch.permute(0, 2, 1)


def af_collate_fn(batch):
    tensors, face_tensors, targets = [], [], []
    if batch[0].audio is None:
        audio = False
        tensors = None
    else:
        audio = True

    for data_collection in batch:
        if audio:
            tensors += [data_collection.audio]
        face_tensors += [data_collection.visual]
        targets += [torch.tensor(data_collection.label)]

    if audio:
        tensors = af_pad_sequence(tensors)
        tensors = tensors.squeeze(1)
    face_tensors = torch.stack(face_tensors)
    targets = torch.stack(targets)

    return DataCollection(visual=face_tensors, audio=tensors, label=targets)


def reg_collate(batch):
    visual, audio, label = [], [], []
    for item in batch:
        visual.append(item.visual)
        audio.append(item.audio)
        label.append(item.label)
    return DataCollection(
        visual=torch.stack(visual),
        audio=torch.stack(audio).squeeze(1),
        label=torch.tensor(label),
    )
