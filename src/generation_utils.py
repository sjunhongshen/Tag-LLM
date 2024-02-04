import inspect
import warnings
from typing import Any, Dict, List, Optional, Union

import torch
import torch.distributed as dist
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import (
    StoppingCriteriaList, validate_stopping_criteria)
from transformers.generation.utils import (GenerationMixin,
                                           GreedySearchDecoderOnlyOutput,
                                           GreedySearchEncoderDecoderOutput,
                                           GreedySearchOutput)
from transformers.utils import ModelOutput, logging

logger = logging.get_logger(__name__)


class TagGenerationMixin(GenerationMixin):
    """Overrides GenerationMixin with special handling for attention masks."""

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
    ) -> Dict[str, Any]:
        # update past_key_values
        model_kwargs["past_key_values"] = self._extract_past_from_model_output(
            outputs, standardize_cache_format=standardize_cache_format
        )

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat(
                [token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1
            )

        if not is_encoder_decoder:
            # update attention mask
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                assert attention_mask.ndim == 2, (
                    "Expected 2d attention mask. This code doesn't work "
                    "with 3D attention masks"
                )
                model_kwargs["attention_mask"] = torch.cat(
                    [
                        attention_mask,
                        attention_mask.new_ones((attention_mask.shape[0], 1)),
                    ],
                    dim=-1,
                )
            if "attention_mask_new" in model_kwargs:
                attention_mask_new = model_kwargs["attention_mask_new"]

                assert attention_mask_new.ndim == 4, "Expected 4D attention mask."
                assert attention_mask_new.shape[1] == 1 and (
                    # Attention mask should either be (B, 1, N, N) (at the
                    # start of decoding) or (B, 1, 1, N) (midway through
                    # decoding, since input ids only considers input for
                    # the next token assuming other values are cached.
                    attention_mask_new.shape[2] == 1
                    or attention_mask_new.shape[2] == attention_mask_new.shape[3]
                ), f"Got attention mask of shape {attention_mask_new.shape}"
                # Here, M is either N or 1.
                last_row = attention_mask_new[
                    :, :, -1:
                ]  # (B, 1, M, N) -> (B, 1, 1, N)
                attention_mask_new = torch.cat(
                    [
                        last_row,  # (B, 1, 1, N)
                        last_row.new_ones((last_row.shape[0], 1, 1, 1)),  # (B, 1, 1, 1)
                    ],
                    dim=-1,
                )  # (B, 1, 1, N) -> (B, 1, 1, N + 1)
                model_kwargs["attention_mask_new"] = attention_mask_new
        else:
            # update decoder attention mask
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                model_kwargs["decoder_attention_mask"] = torch.cat(
                    [
                        decoder_attention_mask,
                        decoder_attention_mask.new_ones(
                            (decoder_attention_mask.shape[0], 1)
                        ),
                    ],
                    dim=-1,
                )

        return model_kwargs

