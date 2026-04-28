import logging
from dataclasses import dataclass
from transformers import ProcessorMixin, AutoProcessor, AutoTokenizer
from src.arguments import DataArguments, ModelArguments
import torch
from qwen_vl_utils import smart_resize, process_vision_info
from PIL import Image
from src.model.processor import LLAVA_NEXT, QWEN2_VL, QWEN2_5_VL, PHI3V, QWEN2_VL_TOKENSELECTION, QWEN2_5_VL_TOKENSELECTION, process_vlm_inputs_fns

from src.utils import print_rank, print_master
from src.model.processor import VLM_IMAGE_TOKENS, VLM_VIDEO_TOKENS
import io
import base64
from PIL import ImageFile
from openai import OpenAI
import random

ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)
PHI_IMAGE_TOKEN_MAX_INPUT_ID = int(1e9)
LLAVA_IMAGE_TOKEN_ID = 32000

GENERATION_PROMPT ='''The above input is a query/candidate for retrieval. Carefully examine and analyze the above input (which may include text, images, videos, 
        or any combination). Identify and describe the key elements present in the input, such as the main topic, important entities, relationships, context, and any notable features or details that contribute to the overall meaning.
        Finally, synthesize your analysis and reflection into a single word or a concise sentence that best captures the essence of the input for retrieval purposes. 
        If the input is a phrase or word, the summary is that word itself.'''

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

@dataclass
class EvalCollator:
    data_args: DataArguments
    model_args: ModelArguments
    processor: ProcessorMixin

    def __call__(self, examples):
        """
        :param examples: qry, qry_image, pos_text, pos_image
        """
        examples = {'text': [e[0] for e in examples], 'images': [e[1] for e in examples]}
        inputs = process_vlm_inputs_fns[self.model_args.model_backbone](examples,
                                                                        processor = self.processor,
                                                                        max_length = self.data_args.max_len)
        inputs['texts'] = examples['text']
        inputs['images'] = examples['images']
        inputs['image_paths'] = [i.filename if hasattr(i, 'filename') else None for i in examples['images']]
        return inputs


@dataclass
class CLIPCollator:
    data_args: DataArguments
    vis_processors: AutoProcessor
    txt_processors: AutoTokenizer

    def __call__(self, examples):
        """
        :param examples: qry, qry_image, pos_text, pos_image
        """
        inputs = self._get_batch_inputs(examples)
        return inputs

    def _get_batch_inputs(self, examples):
        input_ids, pixel_values, attention_mask = [], [], []
        image_exist, text_exist = False, False
        for example in examples:
            text, image = example
            if image is not None:
                if image.mode == 'L':
                    image = image.convert('RGB')
                image_inputs = self.vis_processors(images=image, return_tensors="pt")
                image_exist = True
                pixel_values.append(image_inputs['pixel_values'])
            if text:
                text_exist = True
            text_inputs = self.txt_processors(text, padding=getattr(self.data_args, "padding", True), max_length=self.data_args.max_len, truncation=True, return_tensors="pt")
            input_ids.append(text_inputs["input_ids"].squeeze(0))
        if text_exist:
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids, batch_first=True, padding_value=self.txt_processors.pad_token_id
            )
            attention_mask = input_ids.ne(self.txt_processors.pad_token_id)
        if image_exist:
            pixel_values = torch.cat(pixel_values, dim=0)
        if text_exist and image_exist:
            assert input_ids.size()[0]==pixel_values.size()[0]
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'pixel_values': pixel_values,
        }

        return inputs


@dataclass
class OpenCLIPCollator:
    data_args: DataArguments
    vis_processors: AutoProcessor
    txt_processors: AutoTokenizer

    def __call__(self, examples):
        """
        :param examples: qry, qry_image, pos_text, pos_image
        """
        inputs = self._get_batch_inputs(examples)
        return inputs

    def _get_batch_inputs(self, examples):
        input_ids, pixel_values, attention_mask = [], [], []
        image_exist, text_exist = False, False
        for example in examples:
            text, image = example
            if image is not None:
                if image.mode == 'L':
                    image = image.convert('RGB')
                image_inputs = self.vis_processors(image).unsqueeze(0)
                image_exist = True
                pixel_values.append(image_inputs)
            if text:
                text_exist = True
            text_inputs = self.txt_processors(text)
            input_ids.append(text_inputs)
        if text_exist:
            input_ids = torch.cat(input_ids, dim=0)
            attention_mask = input_ids.ne(0)
        if image_exist:
            pixel_values = torch.cat(pixel_values, dim=0)
        if text_exist and image_exist:
            assert input_ids.size()[0]==pixel_values.size()[0]
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'pixel_values': pixel_values,
        }

        return inputs



@dataclass
class MultimodalEvalDataCollator:
    processor: ProcessorMixin
    model_args: ModelArguments
    data_args: DataArguments
    encode_side: str

    def _get_batch_inputs(self, batch, text_keyname, image_keyname):
        vlm_image_token, vlm_video_token = VLM_IMAGE_TOKENS[self.model_args.model_backbone], VLM_VIDEO_TOKENS[self.model_args.model_backbone]
        texts, visual_inputs = [], []
        messages = []
        for example in batch:
            if example is None or not example:
                text, visual_input = '  ', None
                messages.append(
                    [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": text + "\n<disc_emb>\n" + GENERATION_PROMPT},
                            ],
                        }
                    ]
                )
            else:
                ex_text, ex_images = example[text_keyname], example[image_keyname]
                # ex_text, ex_images could be one single pair from the query side or a list of pairs from the candidates side
                has_image = isinstance(ex_images, dict) or (isinstance(ex_images, list) and all(isinstance(item, dict) for item in ex_images))
                if has_image:
                    for text, raw_images in zip(ex_text, ex_images):
                        visual_input = []
                        visual_paths = []
                        assert 'resolutions' in raw_images, "we need len(raw_images['resolutions']) to determine the number of images, set it a list of None of for cases that no resizing is needed"
                        num_images = len(raw_images['paths'])
                        for image_idx in range(num_images):
                            bytes = raw_images['bytes'][image_idx] if 'bytes' in raw_images else None
                            path = raw_images['paths'][image_idx] if 'paths' in raw_images else None
                            image_resolution = raw_images['resolutions'][image_idx] if 'resolutions' in raw_images else None
                            if bytes is None and path is None:
                                image = None
                            elif bytes is not None:
                                # vidore, image inputs are already bytes
                                image = Image.open(io.BytesIO(bytes))
                            elif path is not None:
                                # mmeb/video datasets, lazy image loading and processing
                                image = Image.open(path)
                            else:
                                print_rank(f"\n{'=' * 50}\nsomething went wrong with a data point from {example['global_dataset_name']}, neither bytes or path is given. \n\t\tquery_text: {example['query_text']}")
                            if not self.data_args.resize_use_processor and image is not None and image_resolution:
                                image = image.resize(image_resolution)
                            if image is not None and (image_resolution is not None and self.data_args.image_decay_factor is not None):
                                assert image_resolution is None, "image_resolution is conflicting with image_decay_factor"
                                assert self.model_args.model_backbone in [QWEN2_VL, QWEN2_5_VL, QWEN2_VL_TOKENSELECTION, QWEN2_5_VL_TOKENSELECTION], "image_decay_factor is only supported for Qwen models"
                                # TODO: this is a hacky way to decay image resolution, need to be refactored
                                max_pixels = max(self.data_args.resize_min_pixels, self.data_args.resize_max_pixels * self.data_args.image_decay_factor ** (num_images - image_idx))
                                width, height = image.size
                                resized_height, resized_width = smart_resize(
                                    height,
                                    width,
                                    min_pixels=self.data_args.resize_min_pixels,
                                    max_pixels=max_pixels,
                                )
                                image = image.resize((resized_width, resized_height))
                            visual_input.append(image)
                            visual_paths.append(path)

                        base64_visual_input = []
                        ## Base64 encoded image
                        try:
                            for idx, img in enumerate(visual_input):
                                img.load()
                                buffer = io.BytesIO()
                                if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                                    img.save(buffer, format='PNG')
                                else:
                                    if img.mode != 'RGB':
                                        img = img.convert('RGB')
                                    img.save(buffer, format='JPEG')
                                img_bytes = buffer.getvalue()
                                base64_visual_input.append(base64.b64encode(img_bytes).decode('utf-8'))
                        except Exception as e:
                            print(f"error in processing {idx}-th image: {e}")
                            print("visual_paths:", visual_paths)


                        # get the random int [0, 3] to decide which openai api to use
                        global_idx = random.randint(0, 3)
                        assist_model="GLM-4.1V-9B-Thinking"
                        openai_api_key = "EMPTY"
                        client = OpenAI(
                            api_key=openai_api_key,
                            base_url=f"http://30.246.243.239:800{global_idx % 4}/v1"
                        )
                        if vlm_image_token in text:
                            if len(visual_paths) == 1:
                                
                                b64_img = encode_image(visual_paths[0])
                                user_content = [
                                    {
                                        "type": "text",
                                        "text": text.replace(vlm_image_token, '').strip() + "\n" + GENERATION_PROMPT,
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image;base64,{b64_img}"
                                        }
                                    },
                                ]
                                msgs = [
                                            {
                                                "role": "user",
                                                "content": user_content,
                                            },
                                        ]
                                try:
                                    response = client.chat.completions.create(
                                                    messages=msgs,
                                                    model=assist_model,
                                                    max_tokens=8192,
                                                    temperature=1.0,
                                                    top_p=None,
                                                    seed=42,
                                                    extra_body={
                                                        "skip_special_tokens": False,
                                                        "repetition_penalty": 1.0,
                                                    },
                                                )
                                    reasoning = response.choices[0].message.content
                                except Exception as e:
                                    print(f"error in openai api call: {e}")
                                    reasoning = ""
                                
                                messages.append(
                                    [
                                        {
                                            "role": "user",
                                            "content": [
                                                {"type": "image", "image": visual_paths[0]},
                                                {"type": "text", "text": text.replace(vlm_image_token, '').strip() + " " + reasoning + "\n<disc_emb>\n" + GENERATION_PROMPT},
                                            ],
                                        }
                                    ]
                                )
                            else:
                                user_content = [
                                                    {
                                                        "type": "text",
                                                        "text": text.replace(vlm_image_token, '').strip() + "\n" + GENERATION_PROMPT,
                                                    }
                                                ]
                                for b64_img in base64_visual_input:
                                    user_content.append(
                                        {
                                            "type": "image_url",
                                            "image_url": {
                                                "url": f"data:image;base64,{b64_img}"
                                            }
                                        }
                                    )
                                msgs = [
                                            {
                                                "role": "user",
                                                "content": user_content,
                                            },
                                        ]
                                try:
                                    response = client.chat.completions.create(
                                                    messages=msgs,
                                                    model=assist_model,
                                                    max_tokens=8192,
                                                    temperature=1.0,
                                                    top_p=None,
                                                    seed=42,
                                                    extra_body={
                                                        "skip_special_tokens": False,
                                                        "repetition_penalty": 1.0,
                                                    },
                                                )
                                    reasoning = response.choices[0].message.content
                                except Exception as e:
                                    print(f"error in openai api call: {e}")
                                    reasoning = ""

                                messages.append(
                                    [
                                        {
                                            "role": "user",
                                            "content": [
                                                {"type": "image", "image": f"data:image;base64,{','.join(base64_visual_input)}" },
                                                {"type": "text", "text": text.replace(vlm_image_token, '').strip() + " " + reasoning + "\n<disc_emb>\n" + GENERATION_PROMPT},
                                            ],
                                        }
                                    ]
                                )
                        
                        elif vlm_video_token in text:
                            base64_frames = []
                            for video_file in visual_paths:
                                with open(video_file, "rb") as video_file:
                                    b64_video = base64.b64encode(video_file.read()).decode('utf-8')
                                    base64_frames.append(b64_video)
                            
                            user_content = [
                                                {
                                                    "type": "text",
                                                    "text": text.replace(vlm_video_token, '').strip() + "\n" + GENERATION_PROMPT,
                                                }
                                            ]
                            for b64_video in base64_frames:
                                user_content.append(
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:video;base64,{b64_video}"
                                        }
                                    }
                                )
                            msgs = [
                                        {
                                            "role": "user",
                                            "content": user_content,
                                        },
                                    ]
                            try:
                                response = client.chat.completions.create(
                                                messages=msgs,
                                                model=assist_model,
                                                max_tokens=8192,
                                                temperature=1.0,
                                                top_p=None,
                                                seed=42,
                                                extra_body={
                                                    "skip_special_tokens": False,
                                                    "repetition_penalty": 1.0,
                                                },
                                            )
                                reasoning = response.choices[0].message.content
                            except Exception as e:
                                print(f"error in openai api call: {e}")
                                reasoning = ""
                            messages.append(
                                [
                                    {
                                        "role": "user",
                                        "content": [
                                            {
                                                "type": "video", 
                                                "video": visual_paths,
                                            },
                                            {"type": "text", "text": text.replace(vlm_video_token, '').strip()+ " " + reasoning + "\n<disc_emb>\n" + GENERATION_PROMPT},
                                        ],
                                    }
                                ]
                            )
                        else:

                            user_content = [
                                                {
                                                    "type": "text",
                                                    "text": text + "\n" + GENERATION_PROMPT,
                                                }
                                            ]
                            msgs = [
                                        {
                                            "role": "user",
                                            "content": user_content,
                                        },
                                    ]
                            try:
                                response = client.chat.completions.create(
                                                messages=msgs,
                                                model=assist_model,
                                                max_tokens=8192,
                                                temperature=1.0,
                                                top_p=None,
                                                seed=42,
                                                extra_body={
                                                    "skip_special_tokens": False,
                                                    "repetition_penalty": 1.0,
                                                },
                                            )
                                reasoning = response.choices[0].message.content
                            except Exception as e:
                                print(f"error in openai api call: {e}")
                                reasoning = ""

                            blank_image = Image.new("RGB", (28, 28), (255, 255, 255))
                            # base64 encoded blank image
                            buffer = io.BytesIO()
                            blank_image.save(buffer, format='JPEG')
                            base64_blank_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
                            messages.append(
                                [
                                    {
                                        "role": "user",
                                        "content": [
                                            {"type": "image", "image": f"data:image;base64,{base64_blank_image}"},
                                            {"type": "text", "text": text.strip() + " " + reasoning + "\n<disc_emb>\n" + GENERATION_PROMPT},
                                        ],
                                    }
                                ]
                            )

                        # text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        # texts.append(text)
                        # visual_inputs.append(visual_input)
                else:
                    # get the random int [0, 3] to decide which openai api to use
                    global_idx = random.randint(0, 3)
                    assist_model="GLM-4.1V-9B-Thinking"
                    openai_api_key = "EMPTY"
                    client = OpenAI(
                        api_key=openai_api_key,
                        base_url=f"http://30.246.243.239:800{global_idx % 4}/v1"
                    )
                    # flatten the list in cases of multiple candidates
                    for text, visual_input in zip(ex_text, ex_images):
                        user_content = [
                                            {
                                                "type": "text",
                                                "text": text + "\n" + GENERATION_PROMPT,
                                            }
                                        ]
                        msgs = [
                                    {
                                        "role": "user",
                                        "content": user_content,
                                    },
                                ]
                        try:
                            response = client.chat.completions.create(
                                            messages=msgs,
                                            model=assist_model,
                                            max_tokens=8192,
                                            temperature=1.0,
                                            top_p=None,
                                            seed=42,
                                            extra_body={
                                                "skip_special_tokens": False,
                                                "repetition_penalty": 1.0,
                                            },
                                        )
                            reasoning = response.choices[0].message.content
                        except Exception as e:
                            print(f"error in openai api call: {e}")
                            reasoning = ""

                        blank_image = Image.new("RGB", (28, 28), (255, 255, 255))
                        # base64 encoded blank image
                        buffer = io.BytesIO()
                        blank_image.save(buffer, format='JPEG')
                        base64_blank_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
                        
                        messages.append (
                            [
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "image", "image": f"data:image;base64,{base64_blank_image}"},
                                        {"type": "text", "text": text.replace(vlm_video_token, '').replace(vlm_image_token, '').strip() + " " + reasoning + "\n<disc_emb>\n" + GENERATION_PROMPT},
                                    ],
                                }
                            ]
                        )
                        # text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        # texts.append(text)
                        # visual_inputs.append(visual_input)
                        # pass

        inputs = {'messages': messages}
        return inputs


    def __call__(self, examples):
        """
        :param examples: 'query_text', 'query_image', 'cand_text', 'cand_image'
        """
        process_fn = process_vlm_inputs_fns[self.model_args.model_backbone]
        if self.encode_side == 'qry':
            assert type(examples[0]['query_text']) == list or type(examples[0]['query_image']) == list, "Expect text/image to be a list, even it only contains a single element (string, dict or None)"
            inputs = self._get_batch_inputs(examples, "query_text", "query_image")
        else:
            assert type(examples[0]['cand_text']) == list or type(examples[0]['cand_image']) == list, "Expect text/image to be a list, even it only contains a single element (string, dict or None)"
            inputs = self._get_batch_inputs(examples, "cand_text", "cand_image")

        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in inputs['messages']
        ]
        image_inputs, video_inputs, video_kwargs = process_vision_info(inputs['messages'], return_video_kwargs=True)
        processed_inputs = self.processor(
                    text=texts,
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                    **video_kwargs,
                )
        
        # judge if video in the conversation
        # for msg in inputs['messages']:

        # image_inputs, video_inputs = process_vision_info(inputs['messages'])
        # processed_inputs = self.processor(
        #             text=texts,
        #             images=image_inputs,
        #             videos=video_inputs,
        #             padding=True,
        #             return_tensors="pt",
        #         )
        # processed_inputs = process_fn(inputs, processor=self.processor, max_length=self.data_args.max_len)

        dataset_infos = [e["dataset_infos"] for e in examples]
        return processed_inputs, dataset_infos
