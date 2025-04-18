from PIL import Image

def concatenate_images_horizontally(image_list:list[Image.Image]):
    if not image_list:
        return None
    
    widths, heights = zip(*(i.size for i in image_list))

    total_width = sum(widths)
    max_height = max(heights)

    new_image = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for img in image_list:
        new_image.paste(img, (x_offset, 0))
        x_offset += img.size[0]

    return new_image

def concatenate_images_vertically(image_list: list[Image.Image]):
    if not image_list:
        return None
    
    widths, heights = zip(*(i.size for i in image_list))

    max_width = max(widths)
    total_height = sum(heights)

    new_image = Image.new('RGB', (max_width, total_height))

    y_offset = 0
    for img in image_list:
        new_image.paste(img, (0, y_offset))
        y_offset += img.size[1]

    return new_image