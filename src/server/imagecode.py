import random, string
from io import BytesIO

from PIL import Image, ImageDraw, ImageFont, ImageFilter


class ImageCode:
    def __init__(self):
        pass

    def rand_color(self):
        red = random.randint(32, 127)
        green = random.randint(32, 127)
        blue = random.randint(32, 127)
        return red, green, blue

    def gen_text(self):
        list = random.sample(string.ascii_letters + string.digits, 4)
        return ''.join(list)

    def draw_verify_code(self):
        code = self.gen_text()
        width,height = 120,50
        image = Image.new('RGB', (width, height), (255, 255, 255))
        font = ImageFont.truetype('src/resources/font/Arial.ttf', 40)
        draw = ImageDraw.Draw(image)
        for i in range(4):
            draw.text((5 + random.randint(-3, 3) + 23 * i, 5 + random.randint(-3, 3)), text=code[i], fill=self.rand_color(), font=font)
        self.draw_lines(draw, 2, width, height)
        return image, code

    def draw_lines(self, draw, num, width, height):
        for i in range(num):
            x1 = random.randint(0, width / 2)
            y1 = random.randint(0, height / 2)
            x2 = random.randint(0, width)
            y2 = random.randint(height / 2, height)
            draw.line(((x1, y1), (x2, y2)), fill="black", width=2)

    def get_code(self):
        image, code = self.draw_verify_code()
        buf = BytesIO()
        image.save(buf, 'jpeg')
        buf_str = buf.getvalue()
        return code,buf_str
