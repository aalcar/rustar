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

# Number of characters in the string between 4 and 10.
lower_bound = 4
upper_bound = 10
num_chars = choice(range(lower_bound, upper_bound + 1))

# Create list of text characters.
text_chars = []
for i in range(num_chars):
    text_chars.append(choice(possible_chars))

# Create captcha text.
captcha_text = ''.join(text_chars)

# Create captcha! assuming you have a captchas folder in the current directory.
captcha = ImageCaptcha(width = 500, height = 280)

captcha.write(captcha_text, f'captchas/{captcha_text}.png')
