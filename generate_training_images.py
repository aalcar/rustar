# ****************************************************************************
# * Make sure you have captcha installed and Pillow updated to atleast 8.0.0 *
# ****************************************************************************

from captcha.image import ImageCaptcha
from random import choice

# Alphanumeric character options.
possible_chars = ['a', 'A', 'b', 'B', 'c', 'C', 'd', 'D', 'e', 'E', 'f', 'F', 'g', 'G', 'h', 'H', 'i', 'I', 'j', 'J',
                  'k', 'K', 'l', 'L', 'm', 'M', 'n', 'N', 'o', 'O', 'p', 'P', 'q', 'Q', 'r', 'R', 's', 'S', 't', 'T',
                  'u', 'U', 'v', 'V', 'w', 'W', 'x', 'X', 'y', 'Y', 'z', 'Z', '0', '1', '2', '3', '4', '5', '6', '7',
                  '8', '9']

IMAGES_PER_LETTER = 1000

# Create captcha text.
for char in possible_chars: 
    captcha_text = char

    # Create captcha!
    for i in range(1, IMAGES_PER_LETTER + 1):
        captcha = ImageCaptcha(width = 500, height = 280)

        captcha.write(captcha_text, f'training/{char}/image{i}.png')