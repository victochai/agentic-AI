import openai
import replicate
import os
from dotenv import load_dotenv
load_dotenv()
from typing import List
from IPython.display import display, Image
import requests


OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI API key not found. Please set the OPENAI_API_KEY in .env")
if not REPLICATE_API_TOKEN:
    raise ValueError("REPLICATE API token not found. Please set the REPLICATE_API_TOKEN in .env")

with open("instructions.txt", "r", encoding="utf-8") as f:
    instructions = f.read()

client = openai.OpenAI(
    api_key=OPENAI_API_KEY
    )


def get_recipe(
        ingredients: List[str],
        instructions: str
        ) -> str:
    """
    This function takes a list of ingredients and instructions, and returns a recipe.
    """
    print("\nThinking...\n")
    prompt = f"Create a recipe using the following ingredients: {', '.join(ingredients)}. "
    messages = [
        {"role": "system", "content": instructions},
        {"role": "user",   "content": prompt}
    ]
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=messages,
        temperature=1.0
    )
    return response.choices[0].message.content


def summarize_for_image(recipe: str) -> str:
    """
    This function takes a recipe string and returns a description suitable for generating an image.
    """
    messages=[
            {"role": "system", "content":"You are a recipe illustrator. Summarize the following recipe into a concise visual description for an AI image generator."},
            {"role": "user", "content": recipe.strip()}
        ]
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=1.0
    )
    return response.choices[0].message.content



def get_random_welcome() -> str:
    """
    This function generates a random welcome message for users providing ingredients.
    """
    messages = [
        {"role": "system", "content": "You are a 40 y.o Chef Jean-Pierre, a French chef with a playful personality."},
        {"role": "user",
            "content": (
                "You need to create a welcome message for users who will provide ingredients for a recipe. "
                "The message should be friendly, humorous, and remind users to "
                "separate ingredients with commas. You can include a joke about cooking or food to make it light-hearted. "
                "Keep it short (2 sentences maximum) and use simple words. \n\n"
            )
        }
    ]
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=1.0,
        n=1
    )
    return response.choices[0].message.content


def generate_image(recipe: str, style: str = "photorealistic") -> str:
    """
    This function takes a recipe string and returns a description suitable for generating an image.
    """
    prompt = summarize_for_image(recipe)
    full_prompt = f"{style} photo of {prompt}"
    output = replicate.run(
        "stability-ai/stable-diffusion:ac732df83cea7fff18b8472768c88ad041fa750ff7682a21affe81863cbe77e4",
        input={
            "width": 768,
            "height": 768,
            "prompt": full_prompt,
            "scheduler": "K_EULER",
            "num_outputs": 1,
            "guidance_scale": 7.5,
            "num_inference_steps": 50
        }
    )
    return output


if __name__ == "__main__":

    ingredients = input(
        get_random_welcome() + "\n\n"
    ).strip().split(",")

    recipe = get_recipe(ingredients, instructions)
    print(recipe)
    image = generate_image(recipe)

    out = image[0]
    url = out.url
    display(Image(url))
    resp = requests.get(url)
    with open("out.png","wb") as f:
        f.write(resp.content)

