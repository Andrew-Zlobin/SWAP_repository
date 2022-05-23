from locale import resetlocale
#from tracemalloc import reset_peak
import telebot
from nn1 import TextPredictor
#from nn1 import buildPhrase
# NeuralNetworkObject

bot = telebot.TeleBot('5358673977:AAFdnblUsaoLOeMwDRTqzSwWzZ0zp6Y2qDc') # @LanguageProcessingBot

@bot.message_handler(commands=["start"])
def start(m, res=False):
    bot.send_message(m.chat.id, 'Привет! Я испралвю твоё предложение и допишу его.')

@bot.message_handler(content_types=["text"])
def handle_text(message):
    textPredictor = TextPredictor()
    # reusltText = NeuralNetworkObject.getResult(message.text)
    res = textPredictor.buildPhrase(message.text)
    #res = buildPhrase(message.text)
    bot.send_message(message.chat.id, res) # bot.send_message(message.chat.id, reusltText)

bot.polling(none_stop=True, interval=0)