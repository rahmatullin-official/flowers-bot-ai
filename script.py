import telebot
import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np

bot = telebot.TeleBot('')  # your bot token
with open("labels.txt", 'r', encoding="utf-8") as file:
    flowers_names = list([i[2:].strip() for i in file])


@bot.message_handler(commands=['start'])
def help_command(message):
    bot.send_message(message.chat.id, "Пожалуйста отправьте фотографию нужного вам цветка")


@bot.message_handler(content_types=['photo'])
def photo(message):
    fileID = message.photo[-1].file_id
    file_info = bot.get_file(fileID)
    downloaded_file = bot.download_file(file_info.file_path)

    src = 'C:/intel_damir/images' + file_info.file_path[7:]  # change to your diretory
    with open(src, 'wb') as new_file:
        new_file.write(downloaded_file)
    with open(src, "rb") as file:
        np.set_printoptions(suppress=True)
        model = tensorflow.keras.models.load_model('keras_model.h5')
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        image = Image.open(file)
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.ANTIALIAS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        data[0] = normalized_image_array
        prediction = model.predict(data)
        output = f'Это {flowers_names[list(prediction[0]).index(max(prediction[0]))]} с вероятностью ' \
                 f'{int(prediction[0][list(prediction[0]).index(max(prediction[0]))] * 100)}%'
        bot.reply_to(message, output)


@bot.message_handler(content_types=['document'])
def handle_docs_photo(message):
    try:
        chat_id = message.chat.id

        file_info = bot.get_file(message.document.file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        src = 'C:/intel_damir/images' + message.document.file_name
        with open(src, 'wb') as new_file:
            new_file.write(downloaded_file)
        with open(src, "rb") as file:
            np.set_printoptions(suppress=True)
            model = tensorflow.keras.models.load_model('keras_model.h5')
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
            image = Image.open(file)
            size = (224, 224)
            image = ImageOps.fit(image, size, Image.ANTIALIAS)
            image_array = np.asarray(image)
            normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
            data[0] = normalized_image_array
            prediction = model.predict(data)
            output = f'Это {flowers_names[list(prediction[0]).index(max(prediction[0]))]} с вероятностью ' \
                     f'{int(prediction[0][list(prediction[0]).index(max(prediction[0]))] * 100)}%'
            bot.reply_to(message, output)
    except Exception:
        bot.reply_to(message, "Упс, что-то пошло не так, попробуйте еще раз")


bot.polling()
