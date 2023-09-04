from PIL import Image


def hide_text_in_image(input_image_path, output_image_path, secret_text):
    # 打开图像
    image = Image.open(input_image_path)
    width, height = image.size

    # 检查秘密文本长度是否超过图像容量
    max_secret_length = (width * height * 3) // 8
    if len(secret_text) > max_secret_length:
        raise ValueError("秘密文本太长，无法嵌入到图像中。")

    # 转换秘密文本为二进制
    secret_binary = ''.join(format(ord(char), '08b') for char in secret_text)

    index = 0
    for y in range(height):
        for x in range(width):
            pixel = list(image.getpixel((x, y)))
            for color_channel in range(3):
                if index < len(secret_binary):
                    pixel[color_channel] = pixel[color_channel] & 254 | int(secret_binary[index])
                    index += 1
            image.putpixel((x, y), tuple(pixel))

    # 保存带水印的图像
    image.save(output_image_path)


# 使用示例
hide_text_in_image("Bque.jpg", "output_image.jpg", "这是一个LSB水印示例。")


