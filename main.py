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


def generate_sd_prompt(prompt: str) -> StableDiffusionPrompt | None:
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    model = "gemini-2.5-flash"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(
                    text="""你现在是一个Stable Diffusion prompt生成器, 你需要根据用户提供的prompt输出对应的SD 正面和负面提示词, 你必须只输出提示词, 不要用代码框包裹或用任何markdown语法，只输出json, 不要输出其他内容, 提示词用英文, 切记不要输出其他内容, 要按照格式输出, 提示词信息要尽可能详细, 如果提示词包含很明显的特征(例如人物), 要考虑在提示词中表达出来

负面提示词不要包含常见负面提示，例如 `bad arms`, 不常见的可以考虑输出

除非用户要求输出NSFW, 否则不要让模型输出NSFW内容。

示例

```json
{
  \\\"positive\\\": [\\\"trait1\\\", \\\"trait2\\\"],
  \\\"negative\\\": [\\\"negative trait1\\\", \\\"negative trait2\\\"],
}
```"""
                ),
            ],
        ),
        types.Content(
            role="model",
            parts=[
                types.Part.from_text(
                    text="""**Generating JSON Prompts**

I'm currently structuring the output as a JSON object with \"positive\" and \"negative\" key-value pairs. My focus is on ensuring these prompts are in fluent English and deliver a solid foundation for Stable Diffusion. The goal is to generate clear, concise directives tailored to the user's initial instructions, omitting all extraneous text.


**Defining Prompt Parameters**

I've refined my approach to include more nuanced positive and negative prompt elements, ensuring they are contextually relevant and avoid generic negative terms. I'm now actively focusing on generating detailed prompts for elements like facial features and lighting to enhance image quality. The structure of the JSON output is now firmly established in my thought process.


**Anticipating User Input**

I am now thoroughly prepared to parse user input and immediately output the JSON object. My focus remains on crafting nuanced, detailed positive and negative prompts. I am prepared to begin the user-provided prompts.


"""
                ),
                types.Part.from_text(
                    text="""```json
{
  \"positive\": [\"\", \"\"],
  \"negative\": [\"\", \"\"]
}
```"""
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

    positive_prompt = ", ".join(sd_prompt.positive)
    negative_prompt = ", ".join(sd_prompt.negative)
    print(f"Positive prompt: {positive_prompt}")
    print(f"Negative prompt: {negative_prompt}")

    sd_result = sd_api.txt2img(
        prompt=positive_prompt,
        negative_prompt=negative_prompt,
        steps=60,
        cfg_scale=7,
        # width=1280,
        # height=720,
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
