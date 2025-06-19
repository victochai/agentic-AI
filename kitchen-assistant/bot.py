from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.helpers import escape_markdown
import os
from dotenv import load_dotenv
from kitchen_assistant import KitchenAssistant
import random
import asyncio
load_dotenv()


if __name__ == "__main__":


    OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
    REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI API key not found. Please set the OPENAI_API_KEY in .env")
    if not REPLICATE_API_TOKEN:
        raise ValueError("REPLICATE API token not found. Please set the REPLICATE_API_TOKEN in .env")
    if not TELEGRAM_TOKEN:
        raise ValueError("TELEGRAM TOKEN key not found. Please set the TELEGRAM_TOKEN in .env")

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

    # 1. Track user states
    user_states = {}

    # 2. Define welcome handler
    async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        user_states[user_id] = "waiting_for_ingredients"
        welcome_message = kitchen_assistant.get_random_welcome()

        await update.message.reply_text(welcome_message)

    # 3. Handle messages depending on user state
    async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id  # Get user ID
        user_input = update.message.text # Get user input
        state = user_states.get(user_id) # Check if this user is currently in a specific state (e.g. "waiting_for_ingredients")
        if state == "waiting_for_ingredients":
            await update.message.reply_text(random.choice(["\nJust a sec...\n", "\nThinking...\n", "\nGive me a moment...\n", "\nCooking up something special...\n"]))
            response = kitchen_assistant.get_answer(user_input)
            await asyncio.to_thread(kitchen_assistant.generate_image, response)
            response = response.replace("<!-- FINAL_IMAGE -->", "")
            await update.message.reply_text(response, parse_mode= 'HTML')
            await update.message.reply_photo(photo=open(kitchen_assistant.save_jpeg_path, 'rb'), caption="Here's your dish!", parse_mode='HTML')
            user_states[user_id] = "done"
            print("Successfully processed user input:", user_input)
        else:
            await update.message.reply_text("Send /start to begin!")

    # 4. Launch bot
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling()
