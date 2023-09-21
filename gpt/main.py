from telegram import Update, ParseMode, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import CallbackContext, CommandHandler, Updater, MessageHandler, CallbackQueryHandler, PollHandler


def help_command(update: Update, context: CallbackContext) -> None:
    update.message.reply_text("Инстракт: Можно ли увидеть воздух?\n\nОтвет: Воздух – это прозрачный газ, без цвета, вкуса и запаха. Как нам объяснили на кафедре теоретической физики в УдГУ, на самом деле воздух увидеть можно, но только при определенных обстоятельствах. Для того, чтобы их создать, надо посмотреть на воздух через другую среду, например, воду.")
    keyboard = []
    for ind in range(1,8):
        keyboard += [[InlineKeyboardButton(ind, callback_data=ind)]]

    reply_markup = InlineKeyboardMarkup(keyboard)

    update.message.reply_text('Поставьте оценку этому инстракту:', reply_markup=reply_markup)

def continue_team_buildup(update: Update, context: CallbackContext) -> None:
    """Parses the CallbackQuery and updates the message text."""
    query = update.callback_query

    query.answer()

    query.edit_message_text(text=f"Ваша оценка для инстракта: {query.data}\n===============================\n\n")

    context.bot.send_message(chat_id=update.effective_chat.id, text="Инстракт: Можно ли увидеть воздух?\n\nОтвет: Воздух нельзя увидеть, но его можно почувствовать. Вывод: Воздух прозрачный, невидимый, бесцветный, не имеет формы.")
    keyboard = []
    for ind in range(1,8):
        keyboard += [[InlineKeyboardButton(ind, callback_data=ind)]]

    reply_markup = InlineKeyboardMarkup(keyboard)

    context.bot.send_message(chat_id=update.effective_chat.id, text='Поставьте оценку этому инстракту:', reply_markup=reply_markup)




with open('token.txt', encoding="utf-8") as f:
    token = f.read()

#print(token)
updater = Updater(token=token, use_context=True)
dispatcher = updater.dispatcher

#logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#                    level=logging.INFO)

dispatcher.add_handler(CommandHandler("start", help_command))
dispatcher.add_handler(CallbackQueryHandler(continue_team_buildup))

updater.start_polling()