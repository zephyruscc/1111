from PIL import Image


def extract_text_from_image(input_image_path):
    # 打开图像
    image = Image.open(input_image_path)
    width, height = image.size

    secret_binary = ""
    for y in range(height):
        for x in range(width):
            pixel = list(image.getpixel((x, y)))
            for color_channel in range(3):
                secret_binary += str(pixel[color_channel] & 1)

    secret_text = ""
    for i in range(0, len(secret_binary), 8):
        byte = secret_binary[i:i + 8]
        secret_text += chr(int(byte, 2))
        if secret_text[-1] == '\x00':
            break

    return secret_text

# 使用示例
extracted_text = extract_text_from_image("output_image.jpg")
print("提取的水印文本:", extracted_text)

