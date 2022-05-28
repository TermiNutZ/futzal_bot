from telegram import Update, ParseMode, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import CallbackContext, CommandHandler, Updater, MessageHandler, CallbackQueryHandler, PollHandler
from collections import deque
import logging

import matplotlib
from matplotlib import pyplot as plt


import pandas as pd
import numpy as np
import math
from collections import defaultdict
import json
import copy

matplotlib.pyplot.switch_backend('Agg') 


df_stat = pd.read_csv("stats.tsv", delimiter="\t")
df_stat["red_sucks"] = df_stat["red_score"] < df_stat["green_score"]

start_mmr = json.load(open("player_mmr.json", "r"))
mmr_list =[start_mmr]

for i, row in df_stat[df_stat["date"] > "2022-04-06"].sort_values("date").iterrows():
    current_mmr = dict(mmr_list[-1])
    current_mmr["date"] = row["date"]

    red_avg = (current_mmr[row["red_player1"]] + current_mmr[row["red_player2"]] + current_mmr[row["red_player3"]] + current_mmr[row["red_player4"]] + current_mmr[row["red_player5"]])/5
    green_avg = (current_mmr[row["green_player1"]] + current_mmr[row["green_player2"]] + current_mmr[row["green_player3"]] + current_mmr[row["green_player4"]] + current_mmr[row["green_player5"]])/5

    prob_red = 1/(1+10**((green_avg-red_avg)/400))

    multiplier_red = (4/(16**prob_red))**math.copysign(1.0, row["red_score"]-row["green_score"])

    mmr_diff = int(20*(row["red_score"]-row["green_score"])*multiplier_red)

    for i in range(1, 6):
        current_mmr[row["red_player{}".format(i)]] += mmr_diff
        current_mmr[row["green_player{}".format(i)]] -= mmr_diff
        
    mmr_list += [current_mmr]
        
mmr_history_df = pd.DataFrame(mmr_list)




class PlayerStats:
    player_name = ""
    
    game_played = 0
    game_won = 0
    game_lost = 0
    game_tie = 0
    
    team_scored = 0
    team_missed = 0
    
    mmr = -1
    
    def __init__(self, name: str, mmr: int = -1):
        self.player_name = name
        self.mmr = mmr
    
    def fill_stats(self, home_score, guest_score):
        self.game_played += 1
        
        self.game_won += int(home_score > guest_score)
        self.game_lost += int(home_score < guest_score)
        self.game_tie += int(home_score == guest_score)
        
        self.team_scored += home_score
        self.team_missed += guest_score
        
    def to_json(self):
        return {
            "Имя": self.player_name,
            "Количество игр": self.game_played,
            "Побед": self.game_won,
            "Поражений": self.game_lost,
            "Разница побед": self.game_won - self.game_lost,
            
            "Забито командой": self.team_scored,
            "Пропущено командой": self.team_missed,
            "Разница мячей командой": self.team_scored - self.team_missed,
            "MMR": self.mmr
        }

    def to_markdown(self):
        #win_loss_diff = self.game_won - self.game_lost
        answer = (f"*Текущий MMR*\t{self.mmr}\n\n*Количество игр*\t{self.game_played}\n*Побед*\t{self.game_won}"
        f"\n*Поражений*\t{self.game_lost}\n\n*Забито командой*\t{self.team_scored}\n*Пропущено командой*\t{self.team_missed}")

        return answer

    def get_mmr_plot(self, mmr_df):
        plt.figure()
        plt.xkcd()
        plt.plot(mmr_history_df["date"], mmr_history_df[self.player_name])
        plt.title(f"История MMR {self.player_name}")
        filename = f'mmr_plots/{self.player_name}_mmr.png'
        plt.savefig(filename)

        return open(filename, "rb")


class TwoTeamsSplitting:
    
    def __init__(self):
        self.min_diff = 100000
        self.min_avg_green = 0
        self.min_avg_red = 0
        self.greens = []
        self.reds = []


def get_pairs(list_of_players):
    sorted_list_of_players = sorted(list(list_of_players))
    result = []
    for i in range(len(sorted_list_of_players)):
        for j in range(i + 1, len(sorted_list_of_players)):
            result += [(sorted_list_of_players[i], sorted_list_of_players[j])]
    return result

all_players = set()

players_stats = dict()

team_pairs_stats = dict()

df_stat["red_player_set"] = ""
df_stat["green_player_set"] = ""
df_stat["match_players_set"] = ""

for i, row in df_stat.iterrows():
    reds = set(row[["red_player{}".format(i) for i in range(1,6)]].unique())
    greens= set(row[["green_player{}".format(i) for i in range(1,6)]].unique())
    match_players = reds.union(greens)
    all_players = all_players.union(match_players)
    
    df_stat.at[i, "red_player_set"] = reds
    df_stat.at[i, "green_player_set"] = greens
    df_stat.at[i, "match_players_set"] = match_players
    
    for player in reds:
        if player not in players_stats:
            players_stats[player] = PlayerStats(player, mmr_list[-1].get(player))
        players_stats[player].fill_stats(row["red_score"], row["green_score"])
        
    for player in greens:
        if player not in players_stats:
            players_stats[player] = PlayerStats(player, mmr_list[-1].get(player))
        players_stats[player].fill_stats(row["green_score"], row["red_score"])
        
    for pair in get_pairs(reds):
        if pair not in team_pairs_stats:
            team_pairs_stats[pair] = PlayerStats(pair)
        team_pairs_stats[pair].fill_stats(row["red_score"], row["green_score"])
    
    for pair in get_pairs(greens):
        if pair not in team_pairs_stats:
            team_pairs_stats[pair] = PlayerStats(pair)
        team_pairs_stats[pair].fill_stats(row["green_score"], row["red_score"])


def get_total_games_count(df):
    return df.shape[0]

def get_red_sucks_games_count(df):
    return df["red_sucks"].sum()

players_array = [val.to_json() for val in players_stats.values()]

players_df = pd.DataFrame(players_array)
players_df = players_df.sort_values(["Количество игр", "Побед", "Разница мячей командой"], ascending=False)


########################

def get_rating(update: Update, context: CallbackContext):
    try:
        logging.log(logging.INFO, "Trying to build rating")
        current_mmr_df = pd.DataFrame(mmr_history_df.iloc[-1].drop("date").sort_values(ascending=False)).reset_index()
        current_mmr_df.columns = ["Игрок", "MMR"]
        current_mmr_df.index += 1 
        out_data = current_mmr_df.to_markdown()

        context.bot.send_message(chat_id=update.effective_chat.id, text="```stats\n{}```".format(out_data), parse_mode=ParseMode.MARKDOWN_V2)
    except:
        context.bot.send_message(chat_id=update.effective_chat.id, text="Бля, че то пошло не так. Попробуй по-другому")

def player_stat(update: Update, context: CallbackContext) -> None:
    try:
        player_name = " ".join(context.args)

        logging.log(logging.INFO, player_name)

        if player_name not in players_stats:
            update.message.reply_text('Не могу найти игрока {}'.format(player_name))
        
        right_player = players_stats[player_name]
        context.bot.send_photo(chat_id=update.effective_chat.id, photo=right_player.get_mmr_plot(mmr_history_df), caption=right_player.to_markdown(), parse_mode=ParseMode.MARKDOWN_V2)
    except:
        context.bot.send_message(chat_id=update.effective_chat.id, text="Бля, че то пошло не так. Попробуй по-другому")

def start_team_buildup(update: Update, context: CallbackContext) -> None:
    """Sends a message with three inline buttons attached."""

    keyboard = []
    for player in list(players_df.sort_values("Количество игр", ascending=False)["Имя"]):
        if player in mmr_list[-1]:
            keyboard += [[InlineKeyboardButton(player, callback_data=player)]]

    reply_markup = InlineKeyboardMarkup(keyboard)
 

    update.message.reply_text('Выбери состав:', reply_markup=reply_markup)

    context.user_data["current_players"] = []


def generate_teams(players, ind):
    mmr_dict = mmr_list[-1]
    sum_v = sum([mmr_dict[player] for player in players])

    teams_list = []

    for i in range(10):
        for j in range(i + 1, 10):
            for k in range(j + 1, 10):
                for x in range(k + 1, 10):
                    for y in range(x + 1, 10):
                        sum_g = mmr_dict[players[i]] + mmr_dict[players[j]] + mmr_dict[players[k]] + mmr_dict[players[x]] + mmr_dict[players[y]]
                        sum_r = sum_v - sum_g
                        
                        # if abs(sum_g-sum_r) < best.min_diff:
                        #     best.min_diff = abs(sum_g-sum_r)
                        #     best.min_avg_green = sum_g/5
                        #     best.min_avg_red = sum_r/5
                        #     best.greens = [players[i], players[j], players[k], players[x], players[y]]
                        #     best.reds = list(set(players).difference(set(best.greens)))

                        best = TwoTeamsSplitting()

                        best.min_diff = abs(sum_g-sum_r)
                        best.min_avg_green = sum_g/5
                        best.min_avg_red = sum_r/5
                        best.greens = [players[i], players[j], players[k], players[x], players[y]]
                        best.reds = list(set(players).difference(set(best.greens)))

                        teams_list += [best]

    teams_list = sorted(teams_list, key=lambda x: x.min_diff)
    
    return teams_list[ind*2]



def continue_team_buildup(update: Update, context: CallbackContext) -> None:
    """Parses the CallbackQuery and updates the message text."""
    query = update.callback_query

    query.answer()

    if "current_players" not in context.user_data or query.data in context.user_data["current_players"]:
        return

    context.user_data["current_players"] += [query.data]

    # context.user_data["current_players"] = ["Евдокимов", "Кисляков", "Станкевич", "Занкин", "Беляев",
    #  "Русейкин", "Имайчев", "Мелешин", "Козлов", "Фролов"]

    if len(context.user_data["current_players"]) < 10:
        current_players_text = ", ".join(context.user_data["current_players"])
        keyboard = []
        for player in list(players_df.sort_values("Количество игр", ascending=False)["Имя"]):
            if player in mmr_list[-1] and player not in context.user_data["current_players"]:
                keyboard += [[InlineKeyboardButton(player, callback_data=player)]]
        
        reply_markup = InlineKeyboardMarkup(keyboard)

        query.edit_message_text(text=f"Выбери состав: {current_players_text}", reply_markup=reply_markup)

    else:
        
        best = generate_teams(context.user_data["current_players"], 0)

        answer_query = ""
        green_string = ", ".join(best.greens)
        red_string = ", ".join(best.reds)
        answer_query += f"Оптимальные составы\n\nКрасные 🔴: {red_string} (Средний MMR = {best.min_avg_red})\nЗеленые 🟢: {green_string} (Средний MMR = {best.min_avg_green})\n\n"

        # green_string = ", ".join(second_best.greens)
        # red_string = ", ".join(second_best.reds)
        # answer_query += f"Оптимальные составы\n\nКрасные 🔴: {red_string} (Средний MMR = {second_best.min_avg_red})\nЗеленые 🟢: {green_string} (Средний MMR = {second_best.min_avg_green})\n\n"

        query.edit_message_text(text=answer_query)

        message = context.bot.send_poll(
            chat_id = update.effective_chat.id,
            question="Нормальные составы?",
            options=[
                "Отлично, играем",
                "Херня, давай следующие"
            ]
        )

        payload = {
            message.poll.id: {
                "chat_id": update.effective_chat.id,
                "message_id": message.message_id,
                "next_team_ind": 1,
                "players": context.user_data["current_players"]
            }
        }
        context.bot_data.update(payload)

def receive_poll_answer(update: Update, context: CallbackContext) -> None:
    answer = update.poll
    try:
        data = context.bot_data[update.poll.id]
    except:
        return
    print(data)

    result_dict = {x["text"]: x["voter_count"] for x in answer["options"]}

    if result_dict["Херня, давай следующие"] >= 7:

        context.bot.stop_poll(
            chat_id=data["chat_id"],
            message_id=int(data["message_id"])
        )

        best = generate_teams(data["players"], data["next_team_ind"])

        answer_query = ""
        green_string = ", ".join(best.greens)
        red_string = ", ".join(best.reds)
        answer_query += f"Ну вот вам еще состав, нравится?\n\nКрасные 🔴: {red_string} (Средний MMR = {best.min_avg_red})\nЗеленые 🟢: {green_string} (Средний MMR = {best.min_avg_green})\n\n"

        context.bot.send_message(
            chat_id=data["chat_id"],
            text=answer_query,
            reply_to_message_id=int(data["message_id"])
        )

        

        message = context.bot.send_poll(
            chat_id = data["chat_id"],
            question="Нормальные составы?",
            options=[
                "Отлично, играем",
                "Херня, давай следующие"
            ]
        )

        payload = {
            message.poll.id: {
                "chat_id": data["chat_id"],
                "message_id": message.message_id,
                "next_team_ind": int(data["next_team_ind"]) + 1,
                "players": data["players"]
            }
        }
        context.bot_data.update(payload)
    elif result_dict["Отлично, играем"] >= 7:
        context.bot.stop_poll(
            chat_id=data["chat_id"],
            message_id=int(data["message_id"])
        )
        
        message = context.bot.send_poll(
            chat_id = data["chat_id"],
            question="Огонь, играем! А теперь ставки на спорт",
            options=[
                "Красные соснут",
                "Красные соснут, но не сегодня"
            ]
        )




def help_command(update: Update, context: CallbackContext) -> None:
    """Displays info on how to use the bot."""
    update.message.reply_text("/get_rating чтобы получить общий рейтинг\n/player_stat [Фамилия] чтобы получить персональную статистику игрока\n/start_team_buildup чтобы получить идеальный состав")

with open('token.txt', encoding="utf-8") as f:
    token = f.read()

updater = Updater(token=token, use_context=True)
dispatcher = updater.dispatcher

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                     level=logging.INFO)


dispatcher.add_handler(CommandHandler('get_rating', get_rating))
dispatcher.add_handler(CommandHandler("player_stat", player_stat))
dispatcher.add_handler(CommandHandler("start_team_buildup", start_team_buildup))
dispatcher.add_handler(CallbackQueryHandler(continue_team_buildup))
dispatcher.add_handler(CommandHandler("help", help_command))
dispatcher.add_handler(CommandHandler("start", help_command))
dispatcher.add_handler(PollHandler(receive_poll_answer))

updater.start_polling()