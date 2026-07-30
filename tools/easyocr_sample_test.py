from PIL import Image, ImageDraw, ImageFont
import tempfile, os

try:
    from easyocr import Reader
except Exception as exc:
    print('EASYOCR_IMPORT_ERROR', exc)
    raise

img = Image.new('RGB', (480,120), color=(255,255,255))
d = ImageDraw.Draw(img)
try:
    font = ImageFont.truetype('arial.ttf', 36)
except Exception:
    font = None

text = 'Hello 123'
d.text((10,30), text, fill=(0,0,0), font=font)
out = os.path.join(tempfile.gettempdir(), 'easyocr_test.png')
img.save(out)
print('IMAGE_SAVED', out)

reader = Reader(['en'], gpu=False)
res = reader.readtext(out)
print('OCR_RESULT', res)
