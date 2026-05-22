#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
import torch.nn.functional as F
import diffusers
import os
from dataclasses import dataclass
from typing import Optional


DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"
        

class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


@dataclass
class RGBTOutput(CausalLMOutputWithPast):
    loss_ce: Optional[torch.FloatTensor] = None
    loss_edit: Optional[torch.FloatTensor] = None

class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.edit_head = EditMapper(d_llm=config.hidden_size)
        # Initialize weights and apply final processing
        self.post_init()



        p2p_path = getattr(config, "p2p_model_path", "../diffusers/ip2p_ema/")

        # self.scheduler = diffusers.DDPMScheduler.from_pretrained(p2p_path, subfolder="scheduler")
        # self.vae       = diffusers.AutoencoderKL.from_pretrained(p2p_path, subfolder="vae")
        # self.unet      = diffusers.UNet2DConditionModel.from_pretrained(p2p_path, subfolder="unet")

        self._p2p_path = getattr(config, "p2p_model_path", "../diffusers/ip2p_ema/")
        self.scheduler = None
        self.vae = None
        self.unet = None
        self._diffusers_ready = False

        # if hasattr(self.unet.config, "sample_size"):
        #     pass  # ok
        # # Ensure eps vs v-pred alignment
        # if hasattr(self.scheduler.config, "prediction_type") and hasattr(self.unet.config, "prediction_type"):
        #     self.scheduler.register_to_config(prediction_type=self.unet.config.prediction_type)

        # keep VAE in fp32 for stability; UNet follows model dtype
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.vae.requires_grad_(False).to(device, dtype=torch.float32)
        # unet_dtype = self.lm_head.weight.dtype if hasattr(self.lm_head, "weight") else torch.float32
        # self.unet.to(device=device, dtype=unet_dtype)
        # if getattr(self.unet.config, "in_channels", 4) != 8:
        #     self.unet.register_to_config(in_channels=8)
        #     with torch.no_grad():
        #         old = self.unet.conv_in
        #         new = nn.Conv2d(8, old.out_channels, old.kernel_size, old.stride, old.padding, bias=(old.bias is not None))
        #         new.weight.zero_()
        #         new.weight[:, :4] = old.weight
        #         if old.bias is not None:
        #             new.bias.copy_(old.bias)
        #         self.unet.conv_in = new
        
        # # Always keep VAE frozen for stability
        # self.vae.requires_grad_(False)
        # if not getattr(config, "unfreeze_unet", False):
        #     self.unet.requires_grad_(False)

        # if getattr(config, "unfreeze_unet", False):
        # # 1) optionally train conv_in (8->channels) to adapt to 8-ch input
        #     if getattr(config, "tune_unet_conv_in", False):
        #         for p in self.unet.conv_in.parameters():
        #             p.requires_grad_(True)
        #     else:
        #         for p in self.unet.conv_in.parameters():
        #             p.requires_grad_(False)

        # # 2) (Optional) attach LoRA to UNet attention blocks
        #     if getattr(config, "unet_lora_r", 0) > 0:
        #         try:
        #             from diffusers.utils import USE_PEFT_BACKEND
        #             from peft import LoraConfig, get_peft_model
        #         except Exception as e:
        #             raise RuntimeError("Install peft to use UNet LoRA") from e

        #         lora_cfg = LoraConfig(r=config.unet_lora_r, lora_alpha=2*config.unet_lora_r,
        #                             lora_dropout=0.0, bias="none",
        #                             target_modules=["to_q", "to_k", "to_v", "to_out.0"])
        #         self.unet.enable_gradient_checkpointing()  # memory
        #         self.unet = get_peft_model(self.unet, lora_cfg)

    def init_diffusion(self, device=None, dtype=None):
        if self._diffusers_ready:
            return
   
        p2p_path = self._p2p_path

        # Avoid meta-tensor path: load on CPU memory
        self.scheduler = diffusers.DDPMScheduler.from_pretrained(p2p_path, subfolder="scheduler")
        self.vae = diffusers.AutoencoderKL.from_pretrained(
            p2p_path, subfolder="vae", torch_dtype=torch.float32
        )
        self.unet = diffusers.UNet2DConditionModel.from_pretrained(
            p2p_path, subfolder="unet"
        )

        # Align prediction type if needed
        if hasattr(self.scheduler.config, "prediction_type") and hasattr(self.unet.config, "prediction_type"):
            self.scheduler.register_to_config(prediction_type=self.unet.config.prediction_type)

        # Ensure 8-channel input for concat(lat_noisy, lat_inp)
        if getattr(self.unet.config, "in_channels", 4) != 8:
            old = self.unet.conv_in
            new = torch.nn.Conv2d(8, old.out_channels, old.kernel_size, old.stride, old.padding, bias=(old.bias is not None))
            with torch.no_grad():
                new.weight.zero_()
                new.weight[:, :4] = old.weight
                if old.bias is not None: new.bias.copy_(old.bias)
            self.unet.conv_in = new

        # Freeze VAE always
        self.vae.requires_grad_(False)

        # Respect your config flags for UNet trainability
        if not getattr(self.config, "unfreeze_unet", False):
            self.unet.requires_grad_(False)
        else:
            # Optional UNet conv_in finetune
            tune = getattr(self.config, "tune_unet_conv_in", False)
            for p in self.unet.conv_in.parameters():
                p.requires_grad_(tune)

            # Optional UNet LoRA
            if getattr(self.config, "unet_lora_r", 0) > 0:
                from peft import LoraConfig, get_peft_model
                lcfg = LoraConfig(
                    r=self.config.unet_lora_r, lora_alpha=2*self.config.unet_lora_r,
                    lora_dropout=0.0, bias="none",
                    target_modules=["to_q", "to_k", "to_v", "to_out.0"]
                )
                self.unet.enable_gradient_checkpointing()
                self.unet = get_peft_model(self.unet, lcfg)

        # Move to device/dtype now (not during meta init)
        if device is None:
            device = next(self.parameters()).device
        if dtype is None:
            dtype = next(self.parameters()).dtype
        self.vae.to(device, dtype=torch.float32)  # keep VAE fp32
        self.unet.to(device, dtype=dtype)

        self._diffusers_ready = True


    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        p2p_inp: Optional[torch.FloatTensor] = None,
        p2p_ans: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if return_dict is None:
            return_dict = True

        # keep a copy because prepare_* may null input_ids
        orig_input_ids = input_ids

        # Prepare multimodal embeds if needed
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes,
            )

        # Forward through base model; allow extra kwargs (e.g., cache_position)
        try:
            outputs = self.model(
                input_ids=None,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=True,
                **kwargs,
            )
        except TypeError:
            safe_kwargs = dict(kwargs)
            for k in ['cache_position', 'cache_positions']:
                if k in safe_kwargs:
                    safe_kwargs.pop(k)
            outputs = self.model(
                input_ids=None,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=True,
                **safe_kwargs,
            )

        hidden_states = outputs[0]                 # [B, T, H]
        logits = self.lm_head(hidden_states)

        # --- Losses ---
        loss = None
        loss_ce = None
        loss_edit = None

        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_ce = nn.CrossEntropyLoss()(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            loss = loss_ce

        # -------------------- Diffusion / edit loss --------------------
        if (labels is not None) and (p2p_inp is not None) and (p2p_ans is not None):
            # use whichever token ids we have
            ids_for_scan = input_ids if input_ids is not None else orig_input_ids
            if ids_for_scan is None:
                if self.training and not getattr(self, "_warned_no_ids_for_edit", False):
                    print("[warn] input_ids are None; skipping loss_edit this step.")
                    self._warned_no_ids_for_edit = True
                return RGBTOutput(
                    loss=loss, logits=logits,
                    past_key_values=outputs.past_key_values,
                    hidden_states=outputs.hidden_states,
                    attentions=outputs.attentions,
                    loss_ce=loss_ce, loss_edit=None,
                )

            # edit token ids
            edit_ids = getattr(self, "edit_token_ids", None) or getattr(self.config, "edit_token_ids", None)
            if edit_ids is None:
                raise RuntimeError("edit_token_ids not set.")
            edit_ids = torch.as_tensor(edit_ids, device=hidden_states.device, dtype=ids_for_scan.dtype)


            # Find contiguous [IMG0..7] span and slice hidden states
            B, T, H = hidden_states.size()
            llm_slices = []
            for i in range(B):
                ids_i = ids_for_scan[i]
                pos = None
                for s in range(0, ids_i.numel() - 7):
                    if torch.equal(ids_i[s:s+8], edit_ids):
                        pos = s
                        break
                if pos is None:
                    pos = max(0, ids_i.numel() - 9)
                llm_slices.append(hidden_states[i, pos:pos+8, :].unsqueeze(0))
            llm_slice = torch.cat(llm_slices, dim=0)  # [B, 8, H]

            # Corresponding token embeddings for [IMG0..7]
            with torch.no_grad():
                emb_tbl = self.get_model().embed_tokens.weight  # [V, H]
                edit_embs = emb_tbl[edit_ids]                   # [8, H]
                edit_embs = edit_embs.unsqueeze(0).repeat(B, 1, 1)  # [B, 8, H]

            # Map to UNet text-cond
            hid_edit = self.edit_head(llm_slice, edit_embs)  # [B, 77, 768]

            # Lazy init diffusers on the right device/dtype
            if not self._diffusers_ready:
                self.init_diffusion(device=hidden_states.device, dtype=hidden_states.dtype)
            dev = hidden_states.device
            llm_dtype = hidden_states.dtype
            self.vae.to(dev, dtype=torch.float32)
            self.unet.to(dev, dtype=hidden_states.dtype)

            with torch.autocast(device_type="cuda", enabled=False):
                p2p_ans_fp32 = p2p_ans.to(device=dev, dtype=torch.float32)
                p2p_inp_fp32 = p2p_inp.to(device=dev, dtype=torch.float32)

                lat_ans = self.vae.encode(p2p_ans_fp32).latent_dist.sample() * self.vae.config.scaling_factor
                lat_inp = self.vae.encode(p2p_inp_fp32).latent_dist.mode()   * self.vae.config.scaling_factor

            # Now match UNet/LLM dtype for the diffusion path
            lat_ans = lat_ans.to(dtype=llm_dtype)
            lat_inp = lat_inp.to(dtype=llm_dtype)

            noise = torch.randn_like(lat_ans)              # same dtype as lat_ans (bf16 now)
            ts = torch.randint(0, self.scheduler.config.num_train_timesteps, (lat_ans.size(0),), device=dev)
            lat_noisy = self.scheduler.add_noise(lat_ans, noise, ts)

            

            # mild dropouts
            prob = torch.rand(B, device=dev)
            hid_null = self.edit_head(torch.zeros_like(llm_slice), edit_embs)
            cond = torch.where((prob < 0.10).view(B, 1, 1), hid_null, hid_edit)
            lat_inp = torch.where(((prob >= 0.05) & (prob < 0.15)).view(B, 1, 1, 1), torch.zeros_like(lat_inp), lat_inp)

            # UNet predicts noise
            unet_in = torch.cat([lat_noisy, lat_inp], dim=1)  # 8-ch
            pred = self.unet(unet_in, ts, cond).sample

            loss_edit = F.mse_loss(pred, noise, reduction="mean")
            w = getattr(self.config, "edit_loss_weight", 0.5)
            loss = (loss if loss is not None else 0.0) + w * loss_edit
        # ---------------------------------------------------------------

        return RGBTOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            loss_ce=loss_ce,
            loss_edit=loss_edit,
        )





    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs


class EditMapper(nn.Module):
    def __init__(self, d_llm=4096, n_edit=8, d_model=512, nhead=4, nlayers=4, d_ff=2048, d_out=768):
        super().__init__()
        self.llm2hid = nn.Linear(d_llm, d_model)
        self.query = nn.Parameter(torch.randn(1, 77, d_model))
        self.mapper = nn.Transformer(
            batch_first=True, norm_first=True,
            d_model=d_model, nhead=nhead,
            num_encoder_layers=nlayers, num_decoder_layers=nlayers,
            dim_feedforward=d_ff, dropout=0.0
        )
        self.hid2feat = nn.Linear(d_model, d_out)

    def forward(self, llm_slice, edit_token_embs):
        # llm_slice: [B, 8, d_llm]; edit_token_embs: [B, 8, d_llm]
        hid = self.llm2hid(llm_slice + edit_token_embs)
        q = self.query.repeat(hid.size(0), 1, 1)
        hid = self.mapper(hid, q)
        return self.hid2feat(hid)  # [B, 77, 768] for SD text-cond size


AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
