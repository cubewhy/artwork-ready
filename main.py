import json
import time
import os
from dataclasses import dataclass

import PIL.PngImagePlugin
import dotenv
import webuiapi
from google import genai
from google.genai import types


@dataclass
class StableDiffusionPrompt:
    positive: tuple[str]
    negative: tuple[str]
    styles: tuple[str]


def generate_sd_prompt(prompt: str) -> StableDiffusionPrompt | None:
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    model = "gemini-2.5-pro"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(
                    text="""你现在是一个Stable Diffusion prompt生成器, 你需要根据用户提供的prompt输出对应的SD 正面和负面提示词, 你必须只输出提示词, 不要用代码框包裹或用任何markdown语法，只输出json, 不要输出其他内容, 提示词用英文, 切记不要输出其他内容, 要按照格式输出, 提示词信息要尽可能详细, 提示词之间使用逗号分割, 如果提示词包含很明显的特征(例如人物), 要考虑在提示词中表达出来

除非用户要求输出NSFW, 否则不要让模型输出NSFW内容。

示例

```json
{
  \"positive\": [\"trait1\", \"trait2\"],
  \"negative\": [\"negative trait1\", \"negative trait2\"],
  \"styles\": [\"style1\", \"style2\"]
}
```"""
                ),
            ],
        ),
        types.Content(
            role="model",
            parts=[
                types.Part.from_text(
                    text="""**Formulating Image Generation Plans**

I've been working on generating prompts for Stable Diffusion. My strategy is to take the user's input and translate it into specific positive and negative prompts. Right now, I'm focusing on creating prompts suitable for game illustration, with a white background as a key requirement. This is a critical first step. I will be sure to expand on the game illustration focus!


**Refining Prompt Generation Strategy**

I've crafted a generic prompt for a fantasy character, designed to be suitable for game illustration with a white background. My focus now is on ensuring the JSON output format and preventing unwanted content. I'm building negative prompts to exclude common AI artifacts, and I'm ready to receive a user-specified prompt. My confidence level is high regarding my ability to fulfill all requirements.


**Devising a Default Prompt**

I'm ready to generate a default prompt while awaiting the user's input. My focus is on creating a generic yet high-quality prompt for a fantasy character, formatted as a JSON object, with positive and negative prompts. This will serve as a starting point and demonstrate my prompt generation capabilities. I'm taking care to include game illustration specifications and a white background. I believe this will meet all the requirements.


**Initiating Prompt Generation Process**

I've formulated a general plan: Receive a user prompt, craft detailed positive and negative prompts for Stable Diffusion, keeping in mind the need for a white background and game illustration style. I will be sure to avoid NSFW content unless specifically requested, and I'll generate the output as a JSON object. I believe I'm ready to receive the user's input.


**Constructing an Initial Example**

I've formulated a generic example prompt for a \"beautiful elf\" based on the established parameters. The process has highlighted a logical dilemma, which is to provide a response as I haven't been given a user prompt but am instructed to solely generate JSON. I'm now crafting the JSON output with positive and negative prompts that encompass game illustration with a white background and avoiding common AI artifacts. I will generate the code block as my output.


**Preparing Demonstration Output**

I've decided to provide a demonstration prompt as I await user input.  My strategy involves generating a JSON object within a code block, containing both positive and negative prompts. The focus will be on a beautiful elven mage in the game illustration style, against a white background, while excluding common AI artifacts. This setup showcases my ability to fulfill all requirements. I'm now implementing the plan.


"""
                ),
                types.Part.from_text(
                    text="""{
  \"positive\": [
    \"masterpiece\",
    \"best quality\",
    \"ultra-detailed\",
    \"game illustration\",
    \"character design\",
    \"concept art\",
    \"full body shot\",
    \"a beautiful female elf mage\",
    \"long silver hair\",
    \"glowing blue eyes\",
    \"wearing ornate robes with celestial patterns\",
    \"holding a crystal staff that emanates a soft light\",
    \"casting a simple spell\",
    \"serene expression\",
    \"dynamic pose\",
    \"perfect anatomy\",
    \"clean lineart\",
    \"vibrant colors\",
    \"cinematic lighting\",
    \"plain white background\",
    \"simple background\",
    \"solid white background\"
  ],
  \"negative\": [
    \"nsfw\",
    \"lowres\",
    \"bad anatomy\",
    \"bad hands\",
    \"text\",
    \"error\",
    \"missing fingers\",
    \"extra digit\",
    \"fewer digits\",
    \"cropped\",
    \"worst quality\",
    \"low quality\",
    \"normal quality\",
    \"jpeg artifacts\",
    \"signature\",
    \"watermark\",
    \"username\",
    \"blurry\",
    \"bad feet\",
    \"mutated hands\",
    \"extra limbs\",
    \"extra arms\",
    \"extra legs\",
    \"malformed limbs\",
    \"fused fingers\",
    \"too many fingers\",
    \"long neck\",
    \"ugly\",
    \"tiling\",
    \"poorly drawn hands\",
    \"poorly drawn feet\",
    \"poorly drawn face\",
    \"out of frame\",
    \"extra tails\",
    \"deformed\",
    \"disfigured\",
    \"bad art\",
    \"beginner\",
    \"amateur\"
  ],
  \"styles\": [
    \"anime style\"
  ]
}
"""
                ),
            ],
        ),
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt),
            ],
        ),
    ]
    tools = [
        types.Tool(google_search=types.GoogleSearch()),
    ]
    generate_content_config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            thinking_budget=-1,
        ),
        tools=tools,
        response_mime_type="text/plain",
    )

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=generate_content_config,
    ).text

    if response is None:
        return None

    if response.startswith("```"):
        response = "\n".join(response.splitlines()[1:-1])

    # parse json
    return StableDiffusionPrompt(**json.loads(response))


def main():
    prompt = input("Prompt > ")
    batch_size = int(input("Batch Size(Default to 1) > ") or "1")

    print("Please stand by...Generating prompt.")

    # send to gemini to generate sd prompt
    sd_prompt = generate_sd_prompt(prompt)
    if sd_prompt is None:
        print("failed to generate prompt")
        return
    # send to sd
    sd_api = webuiapi.WebUIApi(
        host=os.environ.get("SD_HOST") or "127.0.0.1",
        port=int(os.environ.get("SD_PORT") or "7860"),
    )

    positive_prompt = " ".join(sd_prompt.positive)
    negative_prompt = " ".join(sd_prompt.negative)
    print(f"Positive prompt: {positive_prompt}")
    print(f"Negative prompt: {negative_prompt}")

    sd_result = sd_api.txt2img(
        prompt=positive_prompt,
        negative_prompt=negative_prompt,
        steps=60,
        cfg_scale=7,
        batch_size=batch_size,
    )
    images: list[PIL.PngImagePlugin.PngImageFile] = sd_result.images

    timestamp = time.time()

    for i, image in enumerate(images):
        path = f"generated/gen-{timestamp}/{i}.png"
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        print(f"Saving {path}")
        image.save(path)


if __name__ == "__main__":
    dotenv.load_dotenv()
    main()
