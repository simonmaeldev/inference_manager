# sd webui txt2img img2img api example
```python
from datetime import datetime
import urllib.request
import base64
import json
import time
import os

webui_server_url = 'http://127.0.0.1:7860'

out_dir = 'api_out'
out_dir_t2i = os.path.join(out_dir, 'txt2img')
out_dir_i2i = os.path.join(out_dir, 'img2img')
os.makedirs(out_dir_t2i, exist_ok=True)
os.makedirs(out_dir_i2i, exist_ok=True)


def timestamp():
    return datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")


def encode_file_to_base64(path):
    with open(path, 'rb') as file:
        return base64.b64encode(file.read()).decode('utf-8')


def decode_and_save_base64(base64_str, save_path):
    with open(save_path, "wb") as file:
        file.write(base64.b64decode(base64_str))


def call_api(api_endpoint, **payload):
    data = json.dumps(payload).encode('utf-8')
    request = urllib.request.Request(
        f'{webui_server_url}/{api_endpoint}',
        headers={'Content-Type': 'application/json'},
        data=data,
    )
    response = urllib.request.urlopen(request)
    return json.loads(response.read().decode('utf-8'))


def call_txt2img_api(**payload):
    response = call_api('sdapi/v1/txt2img', **payload)
    for index, image in enumerate(response.get('images')):
        save_path = os.path.join(out_dir_t2i, f'txt2img-{timestamp()}-{index}.png')
        decode_and_save_base64(image, save_path)


def call_img2img_api(**payload):
    response = call_api('sdapi/v1/img2img', **payload)
    for index, image in enumerate(response.get('images')):
        save_path = os.path.join(out_dir_i2i, f'img2img-{timestamp()}-{index}.png')
        decode_and_save_base64(image, save_path)


if __name__ == '__main__':
    payload = {
        "prompt": "masterpiece, (best quality:1.1), 1girl <lora:lora_model:1>",  # extra networks also in prompts
        "negative_prompt": "",
        "seed": 1,
        "steps": 20,
        "width": 512,
        "height": 512,
        "cfg_scale": 7,
        "sampler_name": "DPM++ 2M",
        "n_iter": 1,
        "batch_size": 1,

        # example args for x/y/z plot
        # "script_name": "x/y/z plot",
        # "script_args": [
        #     1,
        #     "10,20",
        #     [],
        #     0,
        #     "",
        #     [],
        #     0,
        #     "",
        #     [],
        #     True,
        #     True,
        #     False,
        #     False,
        #     0,
        #     False
        # ],

        # example args for Refiner and ControlNet
        # "alwayson_scripts": {
        #     "ControlNet": {
        #         "args": [
        #             {
        #                 "batch_images": "",
        #                 "control_mode": "Balanced",
        #                 "enabled": True,
        #                 "guidance_end": 1,
        #                 "guidance_start": 0,
        #                 "image": {
        #                     "image": encode_file_to_base64(r"B:\path\to\control\img.png"),
        #                     "mask": None  # base64, None when not need
        #                 },
        #                 "input_mode": "simple",
        #                 "is_ui": True,
        #                 "loopback": False,
        #                 "low_vram": False,
        #                 "model": "control_v11p_sd15_canny [d14c016b]",
        #                 "module": "canny",
        #                 "output_dir": "",
        #                 "pixel_perfect": False,
        #                 "processor_res": 512,
        #                 "resize_mode": "Crop and Resize",
        #                 "threshold_a": 100,
        #                 "threshold_b": 200,
        #                 "weight": 1
        #             }
        #         ]
        #     },
        #     "Refiner": {
        #         "args": [
        #             True,
        #             "sd_xl_refiner_1.0",
        #             0.5
        #         ]
        #     }
        # },
        # "enable_hr": True,
        # "hr_upscaler": "R-ESRGAN 4x+ Anime6B",
        # "hr_scale": 2,
        # "denoising_strength": 0.5,
        # "styles": ['style 1', 'style 2'],
        # "override_settings": {
        #     'sd_model_checkpoint': "sd_xl_base_1.0",  # this can use to switch sd model
        # },
    }
    call_txt2img_api(**payload)

    init_images = [
        encode_file_to_base64(r"B:\path\to\img_1.png"),
        # encode_file_to_base64(r"B:\path\to\img_2.png"),
        # "https://image.can/also/be/a/http/url.png",
    ]

    batch_size = 2
    payload = {
        "prompt": "1girl, blue hair",
        "seed": 1,
        "steps": 20,
        "width": 512,
        "height": 512,
        "denoising_strength": 0.5,
        "n_iter": 1,
        "init_images": init_images,
        "batch_size": batch_size if len(init_images) == 1 else len(init_images),
        # "mask": encode_file_to_base64(r"B:\path\to\mask.png")
    }
    # if len(init_images) > 1 then batch_size should be == len(init_images)
    # else if len(init_images) == 1 then batch_size can be any value int >= 1
    call_img2img_api(**payload)

    # there exist a useful extension that allows converting of webui calls to api payload
    # particularly useful when you wish setup arguments of extensions and scripts
    # https://github.com/huchenlei/sd-webui-api-payload-display
```

# llama cpp text inference
```python
from llama_cpp import Llama

QWEN_VL_PATH = 'C:/Users/ApprenTyr/.lmstudio/models/openfree/Qwen2.5-VL-32B-Instruct-Q4_K_M-GGUF/qwen2.5-vl-32b-instruct-q4_k_m.gguf'

#basic usage
def basic_usage():
    model = Llama(model_path=QWEN_VL_PATH)
    res = model("The quick brown fox jumps ", stop=["."])
    print(res["choices"][0]["text"])
    # the lazy dog


# loading a chat model
def load_chat_model():
    model = Llama(model_path=QWEN_VL_PATH, chat_format="llama-2")
    print(model.create_chat_completion(
        messages=[{
            "role": "user",
            "content": "what is the meaning of life?"
        }]
    ))

# High level API
# Below is a short example demonstrating how to use the high-level API to for basic text completion:
def high_level_api():
    llm = Llama(
        model_path=QWEN_VL_PATH,
        n_gpu_layers=-1, # Uncomment to use GPU acceleration
        #seed=1337, # Uncomment to set a specific seed
        n_ctx=100000, # Uncomment to increase the context window
        chat_format="llama-2"
    )
    output = llm(
        "What's the reference for 42?", # Prompt
        #max_tokens=32, # Generate up to 32 tokens, set to None to generate up to the end of the context window
        #stop=["Q:", "\n"], # Stop generating just before the model would generate a new question
        #echo=True # Echo the prompt back in the output
    ) # Generate a completion, can also call create_completion
    print(output)
    print(output["choices"][0]["text"])

    # By default llama-cpp-python generates completions in an OpenAI compatible format:
    # {
    # "id": "cmpl-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
    # "object": "text_completion",
    # "created": 1679561337,
    # "model": "./models/7B/llama-model.gguf",
    # "choices": [
    #     {
    #     "text": "Q: Name the planets in the solar system? A: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune and Pluto.",
    #     "index": 0,
    #     "logprobs": None,
    #     "finish_reason": "stop"
    #     }
    # ],
    # "usage": {
    #     "prompt_tokens": 14,
    #     "completion_tokens": 28,
    #     "total_tokens": 42
    # }
    # }

# <create_chat_completion documentation>
# create_chat_completion(messages, functions=None, function_call=None, tools=None, tool_choice=None, temperature=0.2, top_p=0.95, top_k=40, min_p=0.05, typical_p=1.0, stream=False, stop=[], seed=None, response_format=None, max_tokens=None, presence_penalty=0.0, frequency_penalty=0.0, repeat_penalty=1.0, tfs_z=1.0, mirostat_mode=0, mirostat_tau=5.0, mirostat_eta=0.1, model=None, logits_processor=None, grammar=None, logit_bias=None, logprobs=None, top_logprobs=None)

# Generate a chat completion from a list of messages.

# Parameters:

#     messages (List[ChatCompletionRequestMessage]) –

#     A list of messages to generate a response for.
#     functions (Optional[List[ChatCompletionFunction]], default: None ) –

#     A list of functions to use for the chat completion.
#     function_call (Optional[ChatCompletionRequestFunctionCall], default: None ) –

#     A function call to use for the chat completion.
#     tools (Optional[List[ChatCompletionTool]], default: None ) –

#     A list of tools to use for the chat completion.
#     tool_choice (Optional[ChatCompletionToolChoiceOption], default: None ) –

#     A tool choice to use for the chat completion.
#     temperature (float, default: 0.2 ) –

#     The temperature to use for sampling.
#     top_p (float, default: 0.95 ) –

#     The top-p value to use for nucleus sampling. Nucleus sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
#     top_k (int, default: 40 ) –

#     The top-k value to use for sampling. Top-K sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
#     min_p (float, default: 0.05 ) –

#     The min-p value to use for minimum p sampling. Minimum P sampling as described in https://github.com/ggerganov/llama.cpp/pull/3841
#     typical_p (float, default: 1.0 ) –

#     The typical-p value to use for sampling. Locally Typical Sampling implementation described in the paper https://arxiv.org/abs/2202.00666.
#     stream (bool, default: False ) –

#     Whether to stream the results.
#     stop (Optional[Union[str, List[str]]], default: [] ) –

#     A list of strings to stop generation when encountered.
#     seed (Optional[int], default: None ) –

#     The seed to use for sampling.
#     response_format (Optional[ChatCompletionRequestResponseFormat], default: None ) –

#     The response format to use for the chat completion. Use { "type": "json_object" } to contstrain output to only valid json.
#     max_tokens (Optional[int], default: None ) –

#     The maximum number of tokens to generate. If max_tokens <= 0 or None, the maximum number of tokens to generate is unlimited and depends on n_ctx.
#     presence_penalty (float, default: 0.0 ) –

#     The penalty to apply to tokens based on their presence in the prompt.
#     frequency_penalty (float, default: 0.0 ) –

#     The penalty to apply to tokens based on their frequency in the prompt.
#     repeat_penalty (float, default: 1.0 ) –

#     The penalty to apply to repeated tokens.
#     tfs_z (float, default: 1.0 ) –

#     The tail-free sampling parameter.
#     mirostat_mode (int, default: 0 ) –

#     The mirostat sampling mode.
#     mirostat_tau (float, default: 5.0 ) –

#     The mirostat sampling tau parameter.
#     mirostat_eta (float, default: 0.1 ) –

#     The mirostat sampling eta parameter.
#     model (Optional[str], default: None ) –

#     The name to use for the model in the completion object.
#     logits_processor (Optional[LogitsProcessorList], default: None ) –

#     A list of logits processors to use.
#     grammar (Optional[LlamaGrammar], default: None ) –

#     A grammar to use.
#     logit_bias (Optional[Dict[int, float]], default: None ) –

#     A logit bias to use.

# Returns:

#     Union[CreateChatCompletionResponse, Iterator[CreateChatCompletionStreamResponse]] –

#     Generated chat completion or a stream of chat completion chunks.
# </create_chat_completion documentation>



# function calling
# todo better example with good description
def function_calling():
    llm = Llama(model_path=QWEN_VL_PATH, chat_format="chatml-function-calling")
    llm.create_chat_completion(
        messages = [
            {
            "role": "system",
            "content": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. The assistant calls functions with appropriate input when necessary"

            },
            {
            "role": "user",
            "content": "Extract Jason is 25 years old"
            }
        ],
        tools=[{
            "type": "function",
            "function": {
            "name": "UserDetail",
            "parameters": {
                "type": "object",
                "title": "UserDetail",
                "properties": {
                "name": {
                    "title": "Name",
                    "type": "string"
                },
                "age": {
                    "title": "Age",
                    "type": "integer"
                }
                },
                "required": [ "name", "age" ]
            }
            }
        }],
        tool_choice={
            "type": "function",
            "function": {
            "name": "UserDetail"
            }
        }
    )

# main
#basic_usage()
#load_chat_model()
high_level_api()
```

# vision img2txt with Qwen/Qwen2.5-VL-32B-Instruct

Here we show a code snippet to show you how to use the chat model with transformers and qwen_vl_utils:

```python
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-32B-Instruct", torch_dtype="auto", device_map="auto"
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-32B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processer
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-32B-Instruct")

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-32B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
```

Multi image inference
```python
# Messages containing multiple images and a text query
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "file:///path/to/image1.jpg"},
            {"type": "image", "image": "file:///path/to/image2.jpg"},
            {"type": "text", "text": "Identify the similarities between these images."},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Inference
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
```

More Usage Tips

For input images, we support local files, base64, and URLs.

```python
# You can directly insert a local file path, a URL, or a base64-encoded image into the position where you want in the text.
## Local file path
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "file:///path/to/your/image.jpg"},
            {"type": "text", "text": "Describe this image."},
        ],
    }
]
## Image URL
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "http://path/to/your/image.jpg"},
            {"type": "text", "text": "Describe this image."},
        ],
    }
]
## Base64 encoded image
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "data:image;base64,/9j/..."},
            {"type": "text", "text": "Describe this image."},
        ],
    }
]
```

Image Resolution for performance boost

The model supports a wide range of resolution inputs. By default, it uses the native resolution for input, but higher resolutions can enhance performance at the cost of more computation. Users can set the minimum and maximum number of pixels to achieve an optimal configuration for their needs, such as a token count range of 256-1280, to balance speed and memory usage.

```python
min_pixels = 256 * 28 * 28
max_pixels = 1280 * 28 * 28
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2.5-VL-32B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels
)
```

Besides, We provide two methods for fine-grained control over the image size input to the model:

    Define min_pixels and max_pixels: Images will be resized to maintain their aspect ratio within the range of min_pixels and max_pixels.
    Specify exact dimensions: Directly set resized_height and resized_width. These values will be rounded to the nearest multiple of 28.

```python
# min_pixels and max_pixels
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "file:///path/to/your/image.jpg",
                "resized_height": 280,
                "resized_width": 420,
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]
# resized_height and resized_width
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "file:///path/to/your/image.jpg",
                "min_pixels": 50176,
                "max_pixels": 50176,
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]
```
