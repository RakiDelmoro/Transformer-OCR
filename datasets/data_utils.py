import os
import cv2 as cv
import random
import numpy as np
import torch
from itertools import takewhile

from PIL import Image, ImageDraw, ImageFont
from constants import CHARS, END_TOKEN, IMAGE_SIZE, PAD_TOKEN, START_TOKEN, MAX_PHRASE_LENGTH

char_to_index = {c:i for i, c in enumerate(CHARS)}
def get_printable(character):
    if character == PAD_TOKEN: return "🔴"
    if character == START_TOKEN: return "🚦"
    if character == END_TOKEN: return "🤚"
    return character

int_to_printable = [get_printable(c) for _, c in enumerate(CHARS)]

encode = lambda text: [char_to_index[c] for c in text]
decode_for_print = lambda tensor: "".join([int_to_printable[int(each_token)]
                                           for each_token in takewhile(lambda x: x != char_to_index[PAD_TOKEN], tensor)])

max_height_by_font_cache = {}

def generate_random_text(length_of_char: int):
    chars_to_generate = CHARS[4:]

    generate = "".join(random.choice(chars_to_generate) for _ in range(random.randint(1, length_of_char)))

    # generate = "".join(random.choice(chars_to_generate) for _ in range(length_of_char))

    return generate

def text_to_image(text, background="white", text_color="black"):
    font_name = random.choice(os.listdir("datasets/.TrainingFonts"))
    # font_name = "12.ttf"

    max_height_and_font = max_height_by_font_cache.get(font_name)  
    
    if max_height_and_font != None:
        max_height, font = max_height_and_font
        font_size = font.size
    else:
        font_size = 0
        max_height = 0
        while max_height <= IMAGE_SIZE[0]:
            font_size = font_size + 1
            font = ImageFont.truetype(f".fonts/{font_name}", font_size)
            _, top, _, bottom = font.getbbox(''.join(CHARS))
            max_height = bottom - top
        max_height_by_font_cache[font_name] = max_height, font
        font_size = font_size -1
        font = ImageFont.truetype(f".fonts/{font_name}", font_size)

    width = font.getlength(text)
    while width > IMAGE_SIZE[1]:
        font_size = font_size - 1
        font = ImageFont.truetype(f".fonts/{font_name}", font_size)
        width = font.getlength(text)

    image = Image.new("RGB", (IMAGE_SIZE[1], IMAGE_SIZE[0]), color=background)
    draw = ImageDraw.Draw(image)
    _, top, _, _ = font.getbbox(text)

    draw.text((0, -top), text, fill=text_color, font=font)

    image = image.convert("L")
    image_np_arr = np.array(image, dtype=np.float32)
    normalized_image = cv.normalize(image_np_arr, None, 0, 1, cv.NORM_MINMAX, cv.CV_32F)

    return image, normalized_image

def text_to_tensor_and_pad_with_zeros(text, max_length=MAX_PHRASE_LENGTH):
    num_padding = max_length - len(text)

    with_tokens = [char_to_index[START_TOKEN]] + encode(text) + [char_to_index[END_TOKEN]] + [char_to_index[PAD_TOKEN]]
    if num_padding != 0:
        with_tokens.extend([char_to_index[PAD_TOKEN]] * num_padding)
        # if num_padding == 0 else + [char_to_index[PAD_TOKEN] for _ in range(num_padding)]


    return torch.tensor(with_tokens, dtype=torch.long)

def label_to_tensor(character):
    return torch.tensor(encode(character), dtype=torch.long)
