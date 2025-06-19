import openai
import replicate
import os
from dotenv import load_dotenv
load_dotenv()
from typing import List
import requests
import json
import random
from typing import Dict, Any
from openai.types.chat.chat_completion import ChatCompletion


class KitchenAssistant():

    def __init__(
            self,
            openai_api_key: str,
            replicate_api_token: str,
            # Paths to system prompts and tools
            system_prompts_path: str = "system_prompts",
            persona_md = "persona.md",
            welcome_md: str = "welcome.md",
            tools_md: str = "tools.md",
            tools_path: str = "./",
            tools_json: str = "tools.json",
            # Paths to save images and md files
            save_path: str = "output",
            save_md: str = "answer.md",
            save_jpeg: str = "final_dish.jpg"
            ):

        self.client = openai.OpenAI(api_key=openai_api_key)
        replicate.Client(api_token=replicate_api_token)

        # Load system prompts and tools
        self.welcome_md = self._load_md(path=os.path.join(system_prompts_path, welcome_md))
        self.persona_md = self._load_md(path=os.path.join(system_prompts_path, persona_md))
        self.tools_md = self._load_md(path=os.path.join(system_prompts_path, tools_md))

        # Load tools from JSOn file
        self.tools = self._load_json(path=os.path.join(tools_path, tools_json))
        os.makedirs(save_path, exist_ok=True)
        self.save_md_path  = os.path.join(save_path, save_md)
        self.save_jpeg_path = os.path.join(save_path, save_jpeg)


    def get_answer(self, user_input: str) -> ChatCompletion:

        """
        This function generates an answer based on the user prompt and a system prompt (persona_md).
        It uses the OpenAI API to generate a response.
        The response is a recipe based on the ingredients provided by the user.
        It also saves the generated recipe to a markdown file.
        Args:
            user_input (str): The input provided by the user, which contains ingredients for the recipe.
        Returns:
            ChatCompletion: The response from the OpenAI API containing the generated answer.
        """

        # Get a response (recipe) from the model based on the user input
        print("User input received:", user_input)
        print(random.choice(["\nJust a sec...\n", "\nThinking...\n", "\nGive me a moment...\n", "\nCooking up something special...\n"]))
        prompt = f"Create a recipe using the ingredients provided in this prompt: {user_input}."
        messages = [
            {"role": "system", "content": self.persona_md},
            {"role": "user",   "content": prompt}
        ]
        response = self.client.chat.completions.create(
            model="gpt-4.1",
            messages=messages,
            temperature=1.0,
            n=1
        )

        # Save the recipe to a markdown file
        markdown_text = response.choices[0].message.content.strip()
        self._save_md(markdown_text=markdown_text)

        return response.choices[0].message.content.strip()


    def generate_image(self, response: ChatCompletion) -> None:
        """
        This function generates images based on the recipe text in the response.
        It checks if the response contains tool calls for image generation.
        If tool calls are present, it generates images using the provided prompts.
        Args:
            response (ChatCompletion): The response from the OpenAI API containing the generated recipe.
        Returns:
            None
        """
        follow_up = self._tool_call(response)
        for call in follow_up.choices[0].message.tool_calls:
            args = json.loads(call.function.arguments)
            if call.function.name == "_generate_image":
                prompt = args.get("prompt", "")
                image_url = self._generate_image(prompt)
                self._download_image(image_url)


    def _save_md(self, markdown_text: str) -> None:
        """
        This function saves the generated markdown text to a file.
        Args:
            markdown_text (str): The markdown text to be saved.
        Returns:
            None
        """
        with open(self.save_md_path, "w", encoding="utf-8") as f:
            f.write(markdown_text)
        print(f"Recipe generated successfully and saved to {self.save_md_path}")


    def _load_json(self, path: str) -> Dict[str, Any]:
        """
        This function loads a JSON file from the given path.
        Args:
            path (str): The path to the JSON file.
        Returns:
            Dict[str, Any]: The loaded JSON data as a dictionary.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"JSON file not found at {path}. Please check the path and try again.")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


    def _load_md(self, path: str) -> str:
        """
        This function loads a markdown file from the given path.
        Args:
            path (str): The path to the markdown file.
        Returns:
            str: The content of the markdown file as a string.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Markdown file not found at {path}. Please check the path and try again.")
        with open(path, "r", encoding="utf-8") as f:
            return f.read()


    def get_random_welcome(self) -> str:
        """
        This function returns a random welcome message for users who are about to provide ingredients.
        It uses the OpenAI API to generate a welcome message based on the welcome_md system prompt.
        Args:
            None
        Returns:
            str: A welcome message for users.
        """
        messages = [
            {"role": "system", "content": self.welcome_md},
            {"role": "user",
                "content": "Write a short welcome message for users who are about to provide ingredients."
            }
        ]
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=1.0,
            n=1
        )
        return response.choices[0].message.content


    def _tool_call(self, response: ChatCompletion) -> ChatCompletion:
        """
        This function checks if the response contains tool calls.
        It generates a follow-up response that includes the tool calls to generate images based on the recipe.
        It does not call the tools directly, but prepares the response for further processing.
        Args:
            response (ChatCompletion): The response from the OpenAI API containing the generated recipe.
        Returns:
            ChatCompletion: The follow-up response that includes the tool calls for image generation.

        """
        follow_up = self.client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": self.tools_md},
                {"role": "user", "content": "Here’s the recipe you just wrote:\n\n" + response + "\n\nCall tools to generate the final dish image and useful step images."}
            ],
            tools=self.tools,
            tool_choice="auto"
        )
        if follow_up.choices[0].message.tool_calls is None or len(follow_up.choices[0].message.tool_calls) == 0:
            raise ValueError("No image generation tools were called. Make sure the recipe includes the necessary placeholders.")
        return follow_up


    def _generate_image(self, prompt: str) -> str:
        """
        This function generates an image based on the provided prompt using the Replicate API.
        Args:
            prompt (str): The prompt for generating the image.
        Returns:
            str: The URL of the generated image.
        """
        if not prompt:
            raise ValueError("Trying to generate an image. Prompt cannot be empty.")
        # Call the Replicate API to generate the image
        output_url = replicate.run(
            "black-forest-labs/flux-1.1-pro-ultra",
            input={
                "raw": True,  # less synthetic, more natural aesthetic
                "prompt": prompt + "\n Realistic, high-quality, detailed, visually appealing.",
                "aspect_ratio": "3:2",
                "output_format": "jpg",
                "safety_tolerance": 3  # 0 --> paranoid, 3 --> relaxed
            }
        )
        return output_url


    def _download_image(self, url: str) -> str:
        """
        This function downloads an image from the given URL and saves it to the specified path.
        Args:
            url (str): The URL of the image to be downloaded.
        Returns:
            None
        """
        response = requests.get(url)
        response.raise_for_status()  # Raises an error if download failed
        with open(self.save_jpeg_path, "wb") as f:
            f.write(response.content)
        print(f"Image downloaded successfully and saved to {self.save_jpeg_path}")


if __name__ == "__main__":

    OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
    REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI API key not found. Please set the OPENAI_API_KEY in .env")
    if not REPLICATE_API_TOKEN:
        raise ValueError("REPLICATE API token not found. Please set the REPLICATE_API_TOKEN in .env")

    kitchen_assistant = KitchenAssistant(
        openai_api_key=OPENAI_API_KEY,
        replicate_api_token=REPLICATE_API_TOKEN,
        system_prompts_path="system_prompts",
        persona_md="persona.md",
        welcome_md="welcome.md",
        tools_md="tools.md",
        tools_path="./",
        tools_json="tools.json",
        save_path="output",
        save_md="answer.md",
        save_jpeg="final_dish.jpg"
    )

    # Get an answer from the kitchen assistant
    answer = kitchen_assistant.get_answer()
    # Generate images based on the answer
    kitchen_assistant.generate_image(answer)
