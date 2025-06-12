import openai
import replicate
import os
from dotenv import load_dotenv
load_dotenv()
from typing import List
from IPython.display import display, Image
import requests
import json
import re
import random


def load_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found at {path}. Please provide a valid path.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_random_welcome(instructions: str) -> str:
    """
    This function generates a random welcome message for users providing ingredients.
    """
    messages = [
        {"role": "system", "content": instructions},
        {"role": "user",
            "content": "Write a short welcome message for users who are about to provide ingredients. No more than 2 sentences, use simple words."
        }
    ]
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=1.0,
        n=1
    )
    return response.choices[0].message.content


def parse_ingredient_input(raw_input: str) -> list:
    # Replace commas with spaces, then split on any whitespace
    words = re.split(r"[,\s]+", raw_input.strip())
    return [w.strip() for w in words if w]


def create_system_prompt(config):
    """
    This function creates a system prompt for the AI model based on the provided configuration.
    """
    final_dish = "Final dish image is REQUIRED. Use " if config['fcalls']['final_dish_image'] else ""
    step_images = "Include images for recipe steps that might be tricky, confusing, or just easier to understand with a visual." if config['fcalls']['step_images'] else ""
    images_rule = "Insert Markdown image placeholders like ![final image](placeholder_final.jpg) or ![step image](placeholder_step_idx.jpg) where appropriate. " \
                  "Do not include real image URLs — only placeholders." \
                  "Insert the final dish image placeholder in the very beginning." \
                  "Insert the step image placeholder immediately after the step description." if config['fcalls']['final_dish_image'] or config['fcalls']['step_images'] else ""
    return f"""
        {config['name']}
        {config['persona']}
        {config['job']}
        Your tone: {config['style']['tone']}
        Humor rules: {config['style']['humor']}
        {final_dish}
        {step_images}
        {images_rule}
        The format of your response should be structured as follows:
        {', '.join(config['formatting']['structure'])}
        You should always follow these rules:
        1. {config['rules']['gen_behaviour']}.
        2. You are allowed to assume that the user has: {', '.join(config['rules']['default_available_ingredients'])}
        3. Maximim number of ingredients: {config['rules']['max_ingredients']}.
        4. Maximim number of steps: {config['rules']['max_steps']}.
        5. Maximum preparation plus cooking time: {config['rules']['max_time']} min.
        6. Maximum difficuluty: {config['rules']['max_difficulty']} out of 5.
        7. Measurement units: {config['rules']['units']}
        If the ingredients are absurd: {config['absurd_behavior']['if_absurd']}
        If they are valid: {config['absurd_behavior']['if_valid']}
        """


def get_recipe(
        ingredients: List[str],
        instructions: str,
        ) -> str:
    """
    This function takes a list of ingredients and instructions, and returns a recipe.
    """
    ## INFO for me: where and how GPT decides whether to use a tool or not:
    # [function][description] + my prompt will lead the model to make a decision.
    # [parameters][prompt][description] is used to generate the prompt for calling the tool.
    # "parameters": {"type": "object"} means JSON object --> dict with string keys and values or a list of dicts with string keys and values.
    print(random.choice(["\nJust a sec...\n", "\nThinking...\n", "\nGive me a moment...\n", "\nCooking up something special...\n"]))
    prompt = f"Create a recipe using the following ingredients: {', '.join(ingredients)}. "
    messages = [
        {"role": "system", "content": instructions},
        {"role": "user",   "content": prompt}
    ]
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=messages,
        temperature=1.0,
        n=1
    )
    return response # Fun fact: If it calls a tool, message.content will be None. Hence I do not call tools here.


def tool_call(response, instructions, tools):

    recipe_text = response.choices[0].message.content.strip()
    follow_up = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": "Here’s the recipe you just wrote:\n\n" + recipe_text + "\n\nCall tools to generate the final dish image and useful step images."}
        ],
        tools=tools,
        tool_choice="auto"
    )

    # Check if tools were called (for now I want it to be mandatory)
    if follow_up.choices[0].message.tool_calls is None or len(follow_up.choices[0].message.tool_calls) == 0:
        raise ValueError("No image generation tools were called. Make sure the recipe includes the necessary placeholders.")

    return follow_up


def generate_image(prompt: str) -> List[str]:
    """
    This function generates an image based on the provided prompt using the Replicate API.
    """
    if not prompt:
        raise ValueError("Trying to generate an image. Prompt cannot be empty.")
    # Call the Replicate API to generate the image
    output = replicate.run(
        "black-forest-labs/flux-1.1-pro-ultra",
        input={
            "raw": True,  # less synthetic, more natural aesthetic
            "prompt": prompt + "\n Realistic, high-quality, detailed, visually appealing.",
            "aspect_ratio": "3:2",
            "output_format": "jpg",
            "safety_tolerance": 3  # 0 --> paranoid, 3 --> relaxed
        }
    )
    return output


def handle_image_gen(response, instructions, tools):
    """
    This function takes a recipe string and returns a description suitable for generating an image.
    """

    follow_up = tool_call(response, instructions, tools=tools)  # Call the tool if needed

    for call in follow_up.choices[0].message.tool_calls:
        if call.function.name == "generate_dish_image":
            print("Generating final dish image...")
            print(f"Prompt: {call.function.arguments}")
            prompt = call.function.arguments["prompt"]["final_dish_description"]
            output = generate_image(prompt=prompt)
    return output


if __name__ == "__main__":

    OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
    REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI API key not found. Please set the OPENAI_API_KEY in .env")
    if not REPLICATE_API_TOKEN:
        raise ValueError("REPLICATE API token not found. Please set the REPLICATE_API_TOKEN in .env")

    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    config = load_json(path="instructions.json") # Load instr config
    instructions = create_system_prompt(config) # Make it a system prompt
    ingredients = input(get_random_welcome(instructions) + "\n\n") # Get ingredients from user input
    ingredients = parse_ingredient_input(ingredients) # Parse the input into a list of ingredients
    tools = load_json(path="tools.json") # Load tools config

    response = get_recipe(ingredients, instructions)
    print(response.choices[0].message.content)  # Print the recipe content

    output = handle_image_gen(response, instructions, tools=tools)  # Generate images if needed

    print(output)
