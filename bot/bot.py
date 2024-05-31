import asyncio
from functools import partial

from environs import Env
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, ContentType, FSInputFile
from aiogram.filters import Command
from model.inference import infernce
from PIL import Image
from run import OPTIONS


env = Env()
env.read_env()

bot = Bot(env('BOT_TOKEN'))
dp = Dispatcher()


@dp.message(Command(commands='start'))
async def start(message: Message):
    await message.answer('Здравствуйте! Я бот, предназначенный для реставрации старых фотографий. '
                         'Если хотите, чтобы ваше изображение было обработано, то просто пришлите его в этот чат.\n\n'
                         'Ваше изображение должно удовлетворять следующим требованиям:\n'
                         '🟢 Размер фотографии должен быть больше 256x256 пикселей;\n'
                         '🟢 Изображение должно быть в формате PNG, JPEG или BMP;\n'
                         '🟢 Ваше изображение должно содержать повреждения, иначе результат '
                         'может быть непредсказуемым.\n\n'
                         '⚡Обработка абсолютно бесплатная и не займет больше двух минут.\n\n'
                         '❗Ваше изображение будет храниться в памяти сервера только во время обработки, '
                         'а после будет автоматически сотрется.')
    

@dp.message(Command(commands='help'))    
async def start(message: Message):
    await message.answer('Для того, чтобы ваше изображение было успешно обработано, оно '
                         'должно удовлетворять следующим требованиям:\n\n'
                         '🟢 Размер фотографии должен быть больше 256x256 пикселей;\n'
                         '🟢 Изображение должно быть в формате PNG, JPEG или BMP;\n'
                         '🟢 Ваше изображение должно содержать повреждения, иначе результат '
                         'может быть непредсказуемым.')


@dp.message(Command(commands='info'))    
async def start(message: Message):
    await message.answer('Я бот, предназначенный для реставрации старых фотографий. '
                         'Ваше изображение должно удовлетворять следующим требованиям:\n\n'
                         '🟢 Размер фотографии должен быть больше 256x256 пикселей;\n'
                         '🟢 Изображение должно быть в формате PNG, JPEG или BMP;\n'
                         '🟢 Ваше изображение должно содержать повреждения, иначе результат '
                         'может быть непредсказуемым.\n\n'
                         '⚡Обработка абсолютно бесплатная и не займет больше двух минут.\n\n'
                         '❗Ваше изображение будет храниться в памяти сервера только во время обработки, '
                         'а после автоматически сотрется.')

@dp.message(F.content_type == ContentType.PHOTO)
async def send_restored_message(message: Message):
    img_id = message.photo[-1].file_id
    img_file = await bot.get_file(file_id=img_id)
    await bot.download_file(img_file.file_path, r'logs\temp\user_img.png')

    img = Image.open(r'logs\temp\user_img.png').convert('RGB')
    img_shape = img.size
    if img_shape[0] >= 256 and img_shape[1] >= 256:
        loop = asyncio.get_event_loop()
        try:
            await message.answer('Реставрация началсь. Пожалуйста, ожидайте.')
            res = await loop.run_in_executor(None, partial(infernce, 
                                                           user_img=img, 
                                                           inference_options=OPTIONS, 
                                                           todo={'clr': 1}))
            if isinstance(res, str):
                await message.answer(res)
            else:
                res.save(r'logs\temp\user_img.png')
                photo = FSInputFile(r'logs\temp\user_img.png')
                await bot.send_photo(chat_id=message.from_user.id, 
                                     photo=photo, 
                                     caption='Обработка фотографии прошла успешно.')
        except Exception as ex:
            print(ex)
            await message.answer('Извините, что-то пошло не так. \n'
                                 'Пожалуйста, проверьте, удовлетворяет ли ваша фотография '
                                 'необходимым требованиям (/help) и попробуйте снова.')
    else:
        await message.answer('К сожалению, размер вашего изображения не подходит. \n'
                             'Минимальный размер фотографии для обработки 256x256 пикселей. Подробнее: /help')


@dp.message()
async def message(message: Message):
    await message.answer('К сожалению, я не могу отвечать на сообщения пользователей. \n'
                         'Пожалуйста, прочитайте информацию обо мне: /info')


async def start():
    await dp.start_polling(bot)
