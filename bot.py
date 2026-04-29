"""
bot.py — Telegram bot entry point
Send a video → receive an AI-edited Short
"""

import os, asyncio, tempfile, logging
from pathlib import Path
from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import ApplicationBuilder, MessageHandler, CallbackQueryHandler, CommandHandler, filters, ContextTypes
from pipeline import run_pipeline

DEFAULT_PREFS = {
    "font":      "Montserrat",
    "size":      "Large",
    "color":     "Yellow",   # pill background colour
    "border":    "Black",    # kept for API compat, unused in pill style
    "animation": "Pop",      # Pop | Bounce | Slide | None  (reserved for future)
    "style":     "Submagic", # Submagic | Minimal | Bold | Neon
}
USER_PREFS = {}
PENDING_VIDEOS = {}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "8577500532:AAGMHY4Va3RaWQTSpfyPanmxxYgHohjUykw")
WORK_DIR  = Path(tempfile.gettempdir()) / "shorts_bot"
WORK_DIR.mkdir(parents=True, exist_ok=True)


# Emoji labels
PILL_ICONS  = {"Yellow": "🟡", "White": "⚪", "Black": "⬛", "Green": "🟢", "Blue": "🔵", "Pink": "🩷"}
STYLE_ICONS = {"Submagic": "✨", "Minimal": "◽", "Bold": "🔲", "Neon": "💡"}
SIZE_ICONS  = {"Small": "S", "Medium": "M", "Large": "L"}

def _settings_text(prefs: dict) -> str:
    pill_icon  = PILL_ICONS.get(prefs["color"], "")
    style_icon = STYLE_ICONS.get(prefs.get("style", "Submagic"), "")
    return (
        "🎬 *Caption Settings* — tap to customise or hit Start:\n\n"
        f"✍️ Font: `{prefs['font']}`  •  Size: `{prefs['size']}`\n"
        f"🟨 Pill Color: {pill_icon} `{prefs['color']}`\n"
        f"💬 Caption Style: {style_icon} `{prefs.get('style','Submagic')}`\n\n"
        "_Key words are auto-detected by AI and shown as solo big captions._"
    )


def _settings_keyboard() -> InlineKeyboardMarkup:
    keyboard = [
        [InlineKeyboardButton("✍️ Font",          callback_data="menu_font"),
         InlineKeyboardButton("📐 Size",           callback_data="menu_size")],
        [InlineKeyboardButton("🟨 Pill Color",     callback_data="menu_color"),
         InlineKeyboardButton("💬 Caption Style",  callback_data="menu_capstyle")],
        [InlineKeyboardButton("↩️ Use Defaults",   callback_data="apply_defaults"),
         InlineKeyboardButton("▶️ Start",           callback_data="start_processing")],
    ]
    return InlineKeyboardMarkup(keyboard)


def _clear_pending(user_id: int):
    pending = PENDING_VIDEOS.pop(user_id, None)
    if not pending:
        return
    for p in [pending.get("input_path"), pending.get("output_path")]:
        if not p:
            continue
        try:
            Path(p).unlink()
        except Exception:
            pass


async def _start_processing(user_id: int, ctx: ContextTypes.DEFAULT_TYPE, chat_id: int):
    pending = PENDING_VIDEOS.get(user_id)
    if not pending:
        await ctx.bot.send_message(chat_id=chat_id, text="No pending video found. Please upload a video first.")
        return

    prefs = USER_PREFS.get(user_id, dict(DEFAULT_PREFS))
    input_path = pending["input_path"]
    output_path = pending["output_path"]

    style_icon = STYLE_ICONS.get(prefs.get("style", "Submagic"), "✨")
    pill_icon  = PILL_ICONS.get(prefs.get("color", "Yellow"), "🟡")
    await ctx.bot.send_message(
        chat_id=chat_id,
        text=(
            f"Got it! Rendering with {style_icon} *{prefs.get('style','Submagic')}* captions "
            f"and {pill_icon} *{prefs['color']}* pill.\n"
            "AI will auto-pick emphasis words and place 1–2 emojis. Takes 1–2 min…"
        ),
        parse_mode="Markdown",
    )

    try:
        result = await asyncio.to_thread(run_pipeline, input_path, output_path, prefs)
        caption = (
            f"Your Short is ready!\n\n"
            f"Style: {result.get('style','auto')}\n"
            f"Hook: {result.get('hook','')}\n"
            f"Cuts: {result.get('n_cuts', 0)} smart cuts\n"
            f"B-roll: {result.get('n_broll', 0)} AI-generated frames\n\n"
            f"Gemma saw: {result.get('scene_description','')}\n"
            f"Visual tone: {result.get('visual_style','')}"
        )
        with open(output_path, "rb") as f:
            await ctx.bot.send_video(chat_id=chat_id, video=f, caption=caption)
    except Exception as e:
        log.exception("Pipeline failed")
        await ctx.bot.send_message(chat_id=chat_id, text=f"Something went wrong: {e}")
    finally:
        _clear_pending(user_id)


async def handle_video(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    video = msg.video or msg.document

    if not video:
        await msg.reply_text("Send me a video file and I'll turn it into a Short!")
        return

    if getattr(video, "file_size", 0) and video.file_size > 20 * 1024 * 1024:
        await msg.reply_text("Too big! Telegram bots can only download files up to 20MB.")
        return

    # Download
    try:
        file = await ctx.bot.get_file(video.file_id)
    except Exception as e:
        await msg.reply_text(f"Could not download video: {e}")
        return

    suffix = Path(video.file_name).suffix if hasattr(video, "file_name") and video.file_name else ".mp4"
    input_path = WORK_DIR / f"{video.file_id}{suffix}"
    await file.download_to_drive(str(input_path))
    log.info(f"Downloaded: {input_path}")

    output_path = WORK_DIR / f"{video.file_id}_short.mp4"

    user_id = update.effective_user.id
    USER_PREFS.setdefault(user_id, dict(DEFAULT_PREFS))

    _clear_pending(user_id)
    PENDING_VIDEOS[user_id] = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "chat_id": msg.chat_id,
    }

    await msg.reply_text(_settings_text(USER_PREFS[user_id]), reply_markup=_settings_keyboard())


async def settings_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id if update.effective_user else None
    prefs = USER_PREFS.get(user_id, dict(DEFAULT_PREFS))
    msg = update.effective_message
    if not msg:
        return
    await msg.reply_text(_settings_text(prefs), reply_markup=_settings_keyboard())

async def button_handler(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    user_id = query.from_user.id
    if user_id not in USER_PREFS:
        USER_PREFS[user_id] = dict(DEFAULT_PREFS)
    
    data = query.data
    
    # Nested Menus
    if data == "menu_font":
        kb = [
            [InlineKeyboardButton("Mm Montserrat", callback_data="font_Montserrat")],
            [InlineKeyboardButton("Aa Arial",       callback_data="font_Arial")],
            [InlineKeyboardButton("Cc Comic Sans",  callback_data="font_Comic")],
            [InlineKeyboardButton("← Back",         callback_data="back_main")],
        ]
        await query.edit_message_text("Select Font Style:", reply_markup=InlineKeyboardMarkup(kb))

    elif data == "menu_size":
        kb = [
            [InlineKeyboardButton("S  Small",  callback_data="size_Small"),
             InlineKeyboardButton("M  Medium", callback_data="size_Medium"),
             InlineKeyboardButton("L  Large",  callback_data="size_Large")],
            [InlineKeyboardButton("← Back",    callback_data="back_main")],
        ]
        await query.edit_message_text("Select Font Size:", reply_markup=InlineKeyboardMarkup(kb))

    elif data == "menu_color":
        kb = [
            [InlineKeyboardButton("🟡 Yellow",  callback_data="color_Yellow"),
             InlineKeyboardButton("⚪ White",   callback_data="color_White")],
            [InlineKeyboardButton("⬛ Black",   callback_data="color_Black"),
             InlineKeyboardButton("🟢 Green",   callback_data="color_Green")],
            [InlineKeyboardButton("🔵 Blue",    callback_data="color_Blue"),
             InlineKeyboardButton("🩷 Pink",    callback_data="color_Pink")],
            [InlineKeyboardButton("← Back",     callback_data="back_main")],
        ]
        await query.edit_message_text("🟨 Select Pill Background Color:\n(this is the box behind your caption text)", reply_markup=InlineKeyboardMarkup(kb))

    elif data == "menu_capstyle":
        kb = [
            [InlineKeyboardButton("✨ Submagic — bold pill, punchy",      callback_data="capstyle_Submagic")],
            [InlineKeyboardButton("◽ Minimal — thinner pill, clean",     callback_data="capstyle_Minimal")],
            [InlineKeyboardButton("🔲 Bold — thick pill, high impact",    callback_data="capstyle_Bold")],
            [InlineKeyboardButton("💡 Neon — glowing shadow, vibrant",    callback_data="capstyle_Neon")],
            [InlineKeyboardButton("← Back",                               callback_data="back_main")],
        ]
        await query.edit_message_text(
            "💬 *Caption Style*\n\n_AI auto-picks emphasis words — shown alone in a larger pill._",
            reply_markup=InlineKeyboardMarkup(kb),
            parse_mode="Markdown",
        )

    elif data == "back_main":
        await query.edit_message_text(
            _settings_text(USER_PREFS[user_id]),
            reply_markup=_settings_keyboard(),
            parse_mode="Markdown",
        )

    elif data == "apply_defaults":
        USER_PREFS[user_id] = dict(DEFAULT_PREFS)
        await query.edit_message_text("↩️ Defaults applied. Starting render…")
        await _start_processing(user_id, ctx, query.message.chat_id)

    elif data == "start_processing":
        await query.edit_message_text("▶️ Settings saved. Starting render…")
        await _start_processing(user_id, ctx, query.message.chat_id)

    # Set values
    elif data.startswith("font_"):
        USER_PREFS[user_id]["font"] = data.split("_", 1)[1]
        await query.edit_message_text(_settings_text(USER_PREFS[user_id]), reply_markup=_settings_keyboard(), parse_mode="Markdown")
    elif data.startswith("size_"):
        USER_PREFS[user_id]["size"] = data.split("_", 1)[1]
        await query.edit_message_text(_settings_text(USER_PREFS[user_id]), reply_markup=_settings_keyboard(), parse_mode="Markdown")
    elif data.startswith("color_"):
        USER_PREFS[user_id]["color"] = data.split("_", 1)[1]
        await query.edit_message_text(_settings_text(USER_PREFS[user_id]), reply_markup=_settings_keyboard(), parse_mode="Markdown")
    elif data.startswith("border_"):
        USER_PREFS[user_id]["border"] = data.split("_", 1)[1]
        await query.edit_message_text(_settings_text(USER_PREFS[user_id]), reply_markup=_settings_keyboard(), parse_mode="Markdown")
    elif data.startswith("capstyle_"):
        USER_PREFS[user_id]["style"] = data.split("_", 1)[1]
        await query.edit_message_text(_settings_text(USER_PREFS[user_id]), reply_markup=_settings_keyboard(), parse_mode="Markdown")


def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler(["settings", "setting"], settings_command))
    app.add_handler(CallbackQueryHandler(button_handler))
    app.add_handler(MessageHandler(filters.VIDEO | filters.Document.VIDEO, handle_video))
    log.info("Bot started. Waiting for videos...")
    app.run_polling()


if __name__ == "__main__":
    main()