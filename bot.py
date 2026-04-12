"""
bot.py — Telegram bot entry point
Send a video → receive an AI-edited Short
"""

import os, asyncio, tempfile, logging
from pathlib import Path
from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import ApplicationBuilder, MessageHandler, CallbackQueryHandler, CommandHandler, filters, ContextTypes
from pipeline import run_pipeline

USER_PREFS = {}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "8577500532:AAGMHY4Va3RaWQTSpfyPanmxxYgHohjUykw")
WORK_DIR  = Path(tempfile.gettempdir()) / "shorts_bot"
WORK_DIR.mkdir(parents=True, exist_ok=True)


async def handle_video(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    video = msg.video or msg.document

    if not video:
        await msg.reply_text("Send me a video file and I'll turn it into a Short!")
        return

    await msg.reply_text(
        "Got it! Processing your video purely based on your /settings...\n"
        "Gemma is analyzing content and Whisper is transcribing. Give me 1-2 mins."
    )

    # Download
    file = await ctx.bot.get_file(video.file_id)
    suffix = Path(video.file_name).suffix if hasattr(video, "file_name") and video.file_name else ".mp4"
    input_path = WORK_DIR / f"{video.file_id}{suffix}"
    await file.download_to_drive(str(input_path))
    log.info(f"Downloaded: {input_path}")

    output_path = WORK_DIR / f"{video.file_id}_short.mp4"

    try:
        user_id = update.effective_user.id
        prefs = USER_PREFS.get(user_id, {"font": "Montserrat", "size": "Large", "color": "Yellow", "border": "Black"})
        
        result = await asyncio.to_thread(run_pipeline, str(input_path), str(output_path), prefs)
        caption = (
            f"Your Short is ready!\n\n"
            f"Style: {result.get('style','auto')}\n"
            f"Hook: {result.get('hook','')}\n"
            f"Cuts: {result.get('n_cuts', 0)} smart cuts\n"
            f"B-roll: {result.get('n_broll', 0)} AI-generated frames\n\n"
            f"Gemma saw: {result.get('scene_description','')}\n"
            f"Visual tone: {result.get('visual_style','')}"
        )
        await msg.reply_video(video=open(output_path, "rb"), caption=caption)
    except Exception as e:
        log.exception("Pipeline failed")
        await msg.reply_text(f"Something went wrong: {e}")
    finally:
        for p in [input_path, output_path]:
            try:
                p.unlink()
            except Exception:
                pass


async def settings_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("Font Style", callback_data="menu_font")],
        [InlineKeyboardButton("Font Size", callback_data="menu_size")],
        [InlineKeyboardButton("Text Color", callback_data="menu_color")],
        [InlineKeyboardButton("Border Color", callback_data="menu_border")],
    ]
    await update.message.reply_text("Configure your subtitle styles:", reply_markup=InlineKeyboardMarkup(keyboard))

async def button_handler(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    user_id = query.from_user.id
    if user_id not in USER_PREFS:
        USER_PREFS[user_id] = {"font": "Montserrat", "size": "Large", "color": "White", "border": "Black"}
    
    data = query.data
    
    # Nested Menus
    if data == "menu_font":
        kb = [[InlineKeyboardButton("Arial", callback_data="font_Arial")],
              [InlineKeyboardButton("Comic Sans", callback_data="font_Comic")],
              [InlineKeyboardButton("Montserrat Bold", callback_data="font_Montserrat")]]
        await query.edit_message_text("Select Font Style:", reply_markup=InlineKeyboardMarkup(kb))
    elif data == "menu_size":
        kb = [[InlineKeyboardButton("Small", callback_data="size_Small"),
               InlineKeyboardButton("Medium", callback_data="size_Medium"),
               InlineKeyboardButton("Large", callback_data="size_Large")]]
        await query.edit_message_text("Select Font Size:", reply_markup=InlineKeyboardMarkup(kb))
    elif data == "menu_color":
        kb = [[InlineKeyboardButton("White", callback_data="color_White"),
               InlineKeyboardButton("Yellow", callback_data="color_Yellow"),
               InlineKeyboardButton("Green", callback_data="color_Green")]]
        await query.edit_message_text("Select Text Color:", reply_markup=InlineKeyboardMarkup(kb))
    elif data == "menu_border":
        kb = [[InlineKeyboardButton("Black", callback_data="border_Black"),
               InlineKeyboardButton("White", callback_data="border_White"),
               InlineKeyboardButton("Red", callback_data="border_Red")]]
        await query.edit_message_text("Select Border Color:", reply_markup=InlineKeyboardMarkup(kb))
        
    # Set Value
    elif data.startswith("font_"):
        fnt = data.split("_")[1]
        USER_PREFS[user_id]["font"] = fnt
        await query.edit_message_text(f"✅ Font set to: {fnt}")
    elif data.startswith("size_"):
        sz = data.split("_")[1]
        USER_PREFS[user_id]["size"] = sz
        await query.edit_message_text(f"✅ Size set to: {sz}")
    elif data.startswith("color_"):
        col = data.split("_")[1]
        USER_PREFS[user_id]["color"] = col
        await query.edit_message_text(f"✅ Text Color set to: {col}")
    elif data.startswith("border_"):
        bdr = data.split("_")[1]
        USER_PREFS[user_id]["border"] = bdr
        await query.edit_message_text(f"✅ Border Color set to: {bdr}")


def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("settings", settings_command))
    app.add_handler(CallbackQueryHandler(button_handler))
    app.add_handler(MessageHandler(filters.VIDEO | filters.Document.VIDEO, handle_video))
    log.info("Bot started. Waiting for videos...")
    app.run_polling()


if __name__ == "__main__":
    main()