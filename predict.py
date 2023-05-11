import time
from typing import Optional
import subprocess

import torch
import os

from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from transformers.models.llama.tokenization_llama import LlamaTokenizer

from tensorizer import TensorDeserializer
from tensorizer.utils import no_init_or_tensor
from collections import OrderedDict
from cog import BasePredictor, ConcatenateIterator, Input, Path


from tensorizer import TensorDeserializer
from tensorizer.utils import no_init_or_tensor

from mplug_owl.configuration_mplug_owl import mPLUG_OwlConfig
from mplug_owl.modeling_mplug_owl import (ImageProcessor,
                                          mPLUG_OwlForConditionalGenerationStream)

from mplug_owl.tokenize_utils import tokenize_prompts


# from config import DEFAULT_MODEL_NAME, DEFAULT_CONFIG_PATH, load_tokenizer, load_tensorizer

TENSORIZER_WEIGHTS_PATH = "gs://replicate-weights/mplug-owl/mplug-owl.tensors"
# TENSORIZER_WEIGHTS_PATH = "model/mplug-owl.tensors"  # path from which we pull weights when there's no COG_WEIGHTS environment variable
# TENSORIZER_WEIGHTS_PATH = None 

DEFAULT_CONFIG_PATH = "model/"
TOKENIZER_PATH = "model/tokenizer.model"

PROMPT_TEMPLATE = """The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
Human: <image>
Human: {input_text}
AI: """

def maybe_download(path):
    if path.startswith("gs://"):
        st = time.time()
        output_path = "/tmp/weights.tensors"
        subprocess.check_call(["gcloud", "storage", "cp", path, output_path])
        print(f"weights downloaded in {time.time() - st}")
        return output_path
    return path


class Predictor(BasePredictor):
    def setup(self, weights: Optional[Path] = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_class = mPLUG_OwlForConditionalGenerationStream

        # set TOKENIZERS_PARALLELISM to false to avoid a warning
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        if weights is not None and weights.name == "weights":
            # bugfix
            weights = None
        if weights is None and TENSORIZER_WEIGHTS_PATH:
            self.model = self.load_tensorizer(
                weights=maybe_download(TENSORIZER_WEIGHTS_PATH), plaid_mode=True, cls=model_class, config_path=DEFAULT_CONFIG_PATH,
            )
        
        elif hasattr(weights, "filename") and "tensors" in weights.filename:
            self.model = self.load_tensorizer(
                weights=weights, plaid_mode=True, cls=model_class, config_path=DEFAULT_CONFIG_PATH,
            )
        elif hasattr(weights, "suffix") and "tensors" in weights.suffix:
            self.model = self.load_tensorizer(
                weights=weights, plaid_mode=True, cls=model_class
            )
        # elif "tensors" in weights:
        #     self.model = self.load_tensorizer(
        #         weights=weights, plaid_mode=True, cls=YieldingMPT
        #     )
        else:
            weights = "./model/"

            self.model = self.load_huggingface_model(weights=weights)

        self.tokenizer = self.load_tokenizer(TOKENIZER_PATH)
        self.image_processor = self.load_image_processor()
    
    def load_tokenizer(self, path):
        tokenizer = LlamaTokenizer(
            path, pad_token='<unk>', add_bos_token=False
            )
        tokenizer.eod_id = tokenizer.eos_token_id
        return tokenizer
    
    def load_image_processor(self):
        return ImageProcessor()

    def load_huggingface_model(self, weights=None):
        raise NotImplementedError()

        # config = AutoConfig.from_pretrained(
        #     weights,
        #     trust_remote_code=True
        # )

        # config.attn_config['attn_impl'] = 'triton'

        # st = time.time()
        # print(f"loading weights from {weights} w/o tensorizer")
        # model = YieldingMPT.from_pretrained(
        #     weights, torch_dtype=torch.bfloat16, trust_remote_code=True
        # )
        # model.to(self.device)
        # print(f"weights loaded in {time.time() - st}")
        # return model
    
    def load_tensorizer(self, weights, plaid_mode, cls, config_path):
        st = time.time()
        print(f"deserializing weights from {weights}")


        config = mPLUG_OwlConfig()

        with no_init_or_tensor():
            model = cls(
                config, 
            )

        deserializer = TensorDeserializer(weights, plaid_mode=plaid_mode)
        deserializer.load_into_module(model)

        print(f"weights loaded in {time.time() - st}")
        model.to(torch.bfloat16)
        model.eval()

        return model

    def predict(
        self,
        prompt: str = Input(description=f"Prompt to send to mPLUG-Owl."),
        img: Path = Input(description='Image to send to mPLUG-Owl.'),
        max_length: int = Input(
            description="Maximum number of tokens to generate. A word is generally 2-3 tokens",
            ge=1,
            default=500,
        ),
        temperature: float = Input(
            description="Adjusts randomness of outputs, greater than 1 is random and 0 is deterministic, 0.75 is a good starting value.",
            ge=0.01,
            le=5,
            default=0.75,
        ),
        top_p: float = Input(
            description="When decoding text, samples from the top p percentage of most likely tokens; lower to ignore less likely tokens",
            ge=0.01,
            le=1.0,
            default=1.0,
        ),
        top_k: int = Input(
            description="When decoding text, samples from the top *k* most likely tokens; lower to ignore less likely tokens. Defaults to 0 (no top-k sampling).",
            ge=1,
            le=500,
            default=1,
        ),
        penalty_alpha: float = Input(
            description="When > 0 and `top_k` > 1, penalizes new tokens based on their similarity to previous tokens. Can help minimize repitition while maintaining semantic coherence. Set to 0 to disable.",
            ge=0.0,
            le=1,
            default=0,
        ),
        repetition_penalty: float = Input(
            description="Penalty for repeated words in generated text; 1 is no penalty, values greater than 1 discourage repetition, less than 1 encourage it.",
            ge=0.01,
            le=5,
            default=1,
        ),
        length_penalty: float = Input(
            description="Increasing the length_penalty parameter above 1.0 will cause the model to favor longer sequences, while decreasing it below 1.0 will cause the model to favor shorter sequences.",
            ge=0.01,
            le=5,
            default=1,
        ),
        no_repeat_ngram_size: int = Input(
            description="If set to int > 0, all ngrams of size no_repeat_ngram_size can only occur once.",
            ge=0,
            default=0,
        ),
        stop_sequence: str = Input(
            description="Generation will hault if this token is produced. Currently, only single token stop sequences are support and it is recommended to use `###` as the stop sequence if you want to control generation termination.",
            default=None,
        ),
        seed: int = Input(
            description="Set seed for reproducible outputs. Set to -1 for random seed.",
            ge=-1,
            default=-1,
        ),
        debug: bool = Input(
            description="provide debugging output in logs", default=False
        ),
    ) -> ConcatenateIterator[str]:
        

        prompts = [PROMPT_TEMPLATE.format(input_text=prompt)]

        images = [str(img)]
        tokens_to_generate = 0
        add_BOS = True

        
        context_tokens_tensor, context_length_tensorm, attention_mask = tokenize_prompts(
            prompts=prompts, 
            tokens_to_generate=tokens_to_generate, 
            add_BOS=add_BOS, 
            tokenizer=self.tokenizer, 
            ignore_dist=True
        )

        images = self.image_processor(images).to(torch.bfloat16)
        images = images.to(self.device)
        context_tokens_tensor = context_tokens_tensor.to(self.device)
        attention_mask = attention_mask.to(self.device)

        # set torch seed
        if seed == -1:
            torch.seed()

        else:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

        with torch.inference_mode():
            first_token_yielded = False
            prev_ids = []
            for output in self.model.generate(
                input_ids=context_tokens_tensor,
                pixel_values=images,
                attention_mask=attention_mask,
                max_length=max_length,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                top_k = top_k, 
                penalty_alpha = penalty_alpha,
            ):
                cur_id = output.item()

                # in order to properly handle spaces, we need to do our own tokenizing. Fun!
                # we're building up a buffer of sub-word / punctuation tokens until we hit a space, and then yielding whole words + punctuation.
                cur_token = self.tokenizer.convert_ids_to_tokens(cur_id)

                # skip initial newline, which this almost always yields. hack - newline id = 13.
                if not first_token_yielded and not prev_ids and cur_id == 187:
                    continue

                # underscore means a space, means we yield previous tokens
                if cur_token.startswith("▁"):  # this is not a standard underscore.
                    # first token
                    if not prev_ids:
                        prev_ids = [cur_id]
                        continue

                    # there are tokens to yield
                    else:
                        token = self.tokenizer.decode(prev_ids) + ' '
                        prev_ids = [cur_id]

                        if not first_token_yielded:
                            # no leading space for first token
                            token = token.strip() + ' '
                            first_token_yielded = True
                        yield token
                                # End token
                elif cur_token == self.tokenizer.eos_token:
                    break
                
                elif stop_sequence and cur_token == stop_sequence:
                    break

                else:
                    prev_ids.append(cur_id)
                    continue

            # remove any special tokens such as </s>
            token = self.tokenizer.decode(prev_ids, skip_special_tokens=True)
            if not first_token_yielded:
                # no leading space for first token
                token = token.strip()
                first_token_yielded = True
            yield token

        if debug:
            print(f"cur memory: {torch.cuda.memory_allocated()}")
            print(f"max allocated: {torch.cuda.max_memory_allocated()}")
            print(f"peak memory: {torch.cuda.max_memory_reserved()}")
