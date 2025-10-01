# worker.py
import os
import cv2
import numpy as np
from io import BytesIO
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters
import logging

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

async def start(update: Update, context) -> None:
    await update.message.reply_text("3x4 uchun rasm yuklang")

async def process_image(update: Update, context) -> None:
    if not update.message.photo:
        await update.message.reply_text("Iltimos, rasm yuboring.")
        return

    try:
        photo_file = await update.message.photo[-1].get_file()
        file_bytes = await photo_file.download_as_bytearray()
        img_array = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            await update.message.reply_text("Rasmni o'qib bo'lmadi.")
            return

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=19)

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                t1 = max(0, x - w // 4)
                t2 = min(img.shape[1], x + int(1.25 * w))
                p1 = max(0, y - h // 2)
                p2 = min(img.shape[0], p1 + int(1.3333 * (t2 - t1)))

                if t2 <= t1 or p2 <= p1:
                    await update.message.reply_text("Yuzni qirqib bo'lmadi.")
                    return

                tayyor = img[p1:p2, t1:t2]
                sisi = cv2.copyMakeBorder(tayyor, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                t2_resized = cv2.resize(sisi, (600, 800))

                photo_canvas = np.ones((3000, 2000, 3), dtype='uint8') * 255
                positions = [
                    (150, 75), (150, 700), (150, 1325),
                    (1000, 75), (1000, 700), (1000, 1325),
                    (1850, 75), (1850, 700), (1850, 1325)
                ]
                for py, px in positions:
                    photo_canvas[py:py+800, px:px+600] = t2_resized

                success, img_encoded = cv2.imencode('.jpg', photo_canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                if not success:
                    await update.message.reply_text("Rasmni saqlab bo'lmadi.")
                    return

                img_byte_array = img_encoded.tobytes()
                await update.message.reply_photo(photo=BytesIO(img_byte_array))
                break
        else:
            await update.message.reply_text("Uzr, odamning yuzini aniqlay olmadik.")
    except Exception as e:
        logging.error(f"Xatolik: {e}")
        await update.message.reply_text("Rasmni qayta ishlashda xatolik yuz berdi.")

def main() -> None:
    TOKEN = os.environ.get("BOT_TOKEN")
    if not TOKEN:
        raise ValueError("BOT_TOKEN muhit o'zgaruvchisi sozlanmagan!")

    application = Application.builder().token(TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.PHOTO & ~filters.COMMAND, process_image))

    print("Bot polling rejimida ishga tushdi...")
    application.run_polling()

if __name__ == '__main__':
    main()
