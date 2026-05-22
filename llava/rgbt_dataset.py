# llava/data/rgb_tir_dataset.py
import os
import copy
import json
from typing import Dict, Sequence, Optional, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as tvt

from llava.constants import (
    IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
)
from llava import conversation as conversation_lib
from llava.mm_utils import tokenizer_image_token

def _safe_replace_rgb_with_tir(path: str) -> str:
    # handle /RGB/ or \RGB\ on any OS
    path2 = path.replace(os.sep + "RGB" + os.sep, os.sep + "TIR" + os.sep)
    path2 = path2.replace("/RGB/", "/TIR/").replace("\\RGB\\", "\\TIR\\")
    return path2

def _expand2square(pil_img: Image.Image, background_color=(0, 0, 0)):
    w, h = pil_img.size
    if w == h:
        return pil_img
    if w > h:
        result = Image.new(pil_img.mode, (w, w), background_color)
        result.paste(pil_img, (0, (w - h) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (h, h), background_color)
        result.paste(pil_img, ((h - w) // 2, 0))
        return result

class RGBTIRSupervisedDataset(Dataset):
    """
    LLaVA-compatible dataset that also (optionally) returns RGB/TIR pairs
    for diffusion training (instructpix2pix). When diffusion pairs are off,
    it behaves like the stock LazySupervisedDataset.

    Keys:
      - Always for LLaVA: input_ids, labels, image (-> collated as `images`)
      - Optional for diffusion: rgb_diffusion, tir_diffusion
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        image_processor,
        image_folder: Optional[str] = None,
        image_aspect_ratio: str = "square",  # {"square","pad"}
        model_max_length: int = 2048,
        # diffusion extras
        return_diffusion_pairs: bool = False,
        diffusion_size: int = 512,
        tir_as_rgb: bool = True,  # True: convert TIR to 3ch; False: keep 1ch ("L")
        mm_use_im_start_end: bool = False,   # NEW
        warn_missing_tir: bool = False,      # optional
    ):
        super().__init__()
        self.data = json.load(open(data_path, "r"))
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.image_folder = image_folder or ""
        self.image_aspect_ratio = image_aspect_ratio
        self.model_max_length = model_max_length
        self.return_diffusion_pairs = return_diffusion_pairs
        self.diffusion_size = diffusion_size
        self.tir_as_rgb = tir_as_rgb
        self.mm_use_im_start_end = mm_use_im_start_end   # NEW
        self.warn_missing_tir = warn_missing_tir


        def make_rgb_transform():
            return tvt.Compose([
                tvt.CenterCrop(min(self.diffusion_size, 10**9)),  # overwritten next line
                tvt.Resize(self.diffusion_size, interpolation=Image.BICUBIC),
                tvt.ToTensor(),
                tvt.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])

        # CenterCrop to exact square first, then Resize
        def square_then_resize(mode_rgb: bool):
            norms = ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) if mode_rgb else ([0.5], [0.5])
            return tvt.Compose([
                # crop to min side square
                tvt.Lambda(lambda im: _expand2square(im, (0, 0, 0)) if im.size[0]!=im.size[1] else im),
                tvt.Resize(self.diffusion_size, interpolation=Image.BICUBIC),
                tvt.ToTensor(),
                tvt.Normalize(*norms),
            ])

        self.diffusion_transform_rgb = square_then_resize(True)
        self.diffusion_transform_tir = square_then_resize(self.tir_as_rgb)
        # diffusion transforms ([-1,1])
        # self.diffusion_transform_rgb = tvt.Compose([
        #     tvt.Resize(self.diffusion_size, interpolation=Image.BICUBIC),
        #     tvt.CenterCrop(self.diffusion_size),
        #     tvt.ToTensor(),
        #     tvt.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        # ])
        # allow 1ch or 3ch for TIR
        # if self.tir_as_rgb:
        #     self.diffusion_transform_tir = tvt.Compose([
        #         tvt.Resize(self.diffusion_size, interpolation=Image.BICUBIC),
        #         tvt.CenterCrop(self.diffusion_size),
        #         tvt.ToTensor(),
        #         tvt.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        #     ])
        # else:
        #     self.diffusion_transform_tir = tvt.Compose([
        #         tvt.Resize(self.diffusion_size, interpolation=Image.BICUBIC),
        #         tvt.CenterCrop(self.diffusion_size),
        #         tvt.ToTensor(),                     # 1 x H x W
        #         tvt.Normalize([0.5], [0.5]),
        #     ])

        # in RGBTIRSupervisedDataset.__init__(...)
        self.inject_edit_tokens = True   # optional flag
        self.edit_span = " ".join([f"<edit_{k}>" for k in range(8)])


    def __len__(self):
        return len(self.data)


    @property
    def lengths(self):
        length_list = []
        for sample in self.data:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.data:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    # add this helper in the class
    def _inject_edit_tokens_into_conversation(self, conv_rounds: list) -> list:
        """
        Ensure the last assistant ('gpt') turn is exactly the 8 edit tokens.
        If there is no assistant turn yet, append one.
        conv_rounds: [{'from': 'human'|'gpt', 'value': str}, ...]
        """
        if not self.inject_edit_tokens:
            return conv_rounds

        # find last assistant turn
        last_gpt = None
        for i in reversed(range(len(conv_rounds))):
            if conv_rounds[i].get("from") == "gpt":
                last_gpt = i
                break

        if last_gpt is None:
            # append a new assistant turn with just the edit tokens
            conv_rounds = conv_rounds + [{"from": "gpt", "value": self.edit_span}]
        else:
            # replace its content with exactly the edit tokens
            conv_rounds = conv_rounds.copy()
            conv_rounds[last_gpt] = {"from": "gpt", "value": self.edit_span}

        return conv_rounds


    
    # --- helpers mirroring your current preprocess path ---
    def _preprocess_multimodal(self, sources):
        use_im_tokens = self.mm_use_im_start_end
        for source in sources:
            for sentence in source:
                if DEFAULT_IMAGE_TOKEN in sentence["value"]:
                    sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
                    sentence["value"] = DEFAULT_IMAGE_TOKEN + "\n" + sentence["value"]
                    sentence["value"] = sentence["value"].strip()
                replace_token = DEFAULT_IMAGE_TOKEN
                if use_im_tokens:
                    replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)
        return sources


    def _preprocess_text(self, sources, has_image: bool):
        # Follow the same path as llava.data.preprocess_v1 / llama_2 depending on conv template
        conv = conversation_lib.default_conversation.copy()
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
        conversations = []

        for i, source in enumerate(sources):
            if roles[source[0]["from"]] != conv.roles[0]:
                source = source[1:]  # skip non-human first
            conv.messages = []
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                assert role == conv.roles[j % 2]
                conv.append_message(role, sentence["value"])
            conversations.append(conv.get_prompt())

        if has_image:
            input_ids = torch.stack(
                [tokenizer_image_token(p, self.tokenizer, return_tensors="pt") for p in conversations], dim=0
            )
        else:
            input_ids = self.tokenizer(
                conversations, return_tensors="pt", padding="longest",
                max_length=self.tokenizer.model_max_length, truncation=True
            ).input_ids

        targets = input_ids.clone()

        # mask human parts (match llava.data.preprocess_v1 / llama_2 implementations)
        if conv.sep_style == conversation_lib.SeparatorStyle.TWO:
            sep = conv.sep + conv.roles[1] + ": "
            for conversation, target in zip(conversations, targets):
                total_len = int(target.ne(self.tokenizer.pad_token_id).sum())
                rounds = conversation.split(conv.sep2)
                cur_len = 1
                target[:cur_len] = IGNORE_INDEX
                for i, rou in enumerate(rounds):
                    if rou == "":
                        break
                    parts = rou.split(sep)
                    if len(parts) != 2:
                        break
                    parts[0] += sep
                    if has_image:
                        round_len = len(tokenizer_image_token(rou, self.tokenizer))
                        instruction_len = len(tokenizer_image_token(parts[0], self.tokenizer)) - 2
                    else:
                        round_len = len(self.tokenizer(rou).input_ids)
                        instruction_len = len(self.tokenizer(parts[0]).input_ids) - 2
                    target[cur_len: cur_len + instruction_len] = IGNORE_INDEX
                    cur_len += round_len
                target[cur_len:] = IGNORE_INDEX
                if cur_len != total_len:
                    target[:] = IGNORE_INDEX  # safe path
        else:
            # fall back to generic masking path (rare given your templates)
            target = targets  # already cloned
        return dict(input_ids=input_ids[0], labels=targets[0])

    def _load_rgb_for_llava(self, pil_img: Image.Image):
        # honor aspect ratio option used by your current code
        if self.image_aspect_ratio == "pad":
            bg = tuple(int(x * 255) for x in self.image_processor.image_mean)
            pil_img = _expand2square(pil_img, bg)
        # LLaVA’s processor returns dict(pixel_values=...)
        return self.image_processor.preprocess(pil_img, return_tensors="pt")["pixel_values"][0]

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        sample = self.data[idx]
        rgb_rel = sample.get("image")
        assert rgb_rel is not None, "Missing 'image' in sample"

        rgb_path = os.path.join(self.image_folder, rgb_rel)
        rgb = Image.open(rgb_path).convert("RGB")

        # paired TIR (optional)
        tir_path = _safe_replace_rgb_with_tir(rgb_path)
        tir_exists = os.path.exists(tir_path)
        if self.return_diffusion_pairs and (not tir_exists) and self.warn_missing_tir:
            print(f"[RGBTIRDataset] Missing TIR for: {rgb_path}")

        # LLaVA image tensor
        image_tensor = self._load_rgb_for_llava(rgb)

        # --------- NEW: build MGIE-style two-turn conversation from your JSON ---------
        # human value is the "instruction"; gpt value is the "expressive" text
        convs = sample["conversations"]
        assert len(convs) >= 2, "Expected at least 2 turns (human, gpt)"

        # take the first human turn as instruction
        human_val = convs[0]["value"]
        # strip a leading <image> (if present) to avoid double image tokens; we’ll insert ours below
        if human_val.lstrip().startswith("<image>"):
            human_val = human_val.lstrip()[len("<image>"):].lstrip()

        # expressive text is the first gpt turn
        expressive = convs[1]["value"].strip()

        # choose the 8 tokens: use the ones you ALREADY added to the tokenizer/model
        EDIT_TOKENS = [f"[IMG{k}]" for k in range(8)]
        edit_span = " ".join(EDIT_TOKENS)

        # build the two messages in MGIE style
        query = f"{human_val}\n{DEFAULT_IMAGE_TOKEN}"            # human: instruction + <image>
        answer = f"{expressive} {edit_span}"                     # gpt: expressive + 8 tokens

        sources = [[
            {"from": "human", "value": query},
            {"from": "gpt",   "value": answer},
        ]]

        # make sure image token formatting matches your config
        sources = self._preprocess_multimodal(copy.deepcopy(sources))
        text_dict = self._preprocess_text(sources, has_image=True)

        # pack outputs for the trainer
        out = {
            "input_ids": text_dict["input_ids"],
            "labels": text_dict["labels"],
            "image": image_tensor,
            "image_sizes": [rgb.size],
        }

        # --------- optional diffusion pairs ---------
        if self.return_diffusion_pairs:
            rgb_for_diff = rgb
            if tir_exists:
                tir_for_diff = Image.open(tir_path).convert("RGB" if self.tir_as_rgb else "L")
            else:
                tir_for_diff = rgb_for_diff.convert("RGB" if self.tir_as_rgb else "L")

            out["p2p_inp"] = self.diffusion_transform_rgb(rgb_for_diff)
            out["p2p_ans"] = self.diffusion_transform_tir(tir_for_diff)

        return out


