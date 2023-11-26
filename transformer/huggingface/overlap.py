from PIL import Image, ImageDraw

def draw_all_bboxes(img, bbox_list):
    """
    Draw all bounding boxes on a PIL image.

    Parameters:
    - image_path: Path to the PIL image.
    - bbox_list: List of bounding boxes, where each bounding box is a tuple (x1, y1, x2, y2).

    Returns:
    - img_with_all_bboxes: PIL image with all bounding boxes drawn on it.
    """
    # Load the PIL image
    # img = Image.open(image_path)

    # Create a copy of the image to draw all bounding boxes
    img_with_all_bboxes = img.copy()
    draw = ImageDraw.Draw(img_with_all_bboxes)

    for bbox in bbox_list:
        # Draw each bounding box on the image
        draw.rectangle(bbox, outline="red", width=2)

    # Display the original image with all bounding boxes
    # img.show()

    # Save the image with all bounding boxes if needed
    # img_with_all_bboxes.show()
    # img_with_all_bboxes.save("image_with_all_bboxes.jpg")

    # Close the images
    # img.close()
    # img_with_all_bboxes.close()

    return img_with_all_bboxes

def check_bbox_overlap(pil_image, bbox_list):
    """
    Check the overlap of a bounding box with the data in a PIL image.

    Parameters:
    - pil_image: PIL image.
    - bbox: Array of bondary boxes

    Returns:
    - overlap_percentage: Percentage of overlap between the bounding box and image data.
    """
    # Load the PIL image
    # pil_image = Image.open(image_path)
    mask = Image.new("L", pil_image.size, 0)
    

    draw = ImageDraw.Draw(mask)

    
    for x1,y1,x2,y2 in bbox_list:
        # Calculate the area of the bounding box
        bbox_area = (x2 - x1) * (y2 - y1)
        # Create a binary mask with 1s inside the bounding box and 0s outside
        draw.rectangle((x1,y1,x2,y2), fill=1)

    intersection_area = sum(mask.getdata())
    combined_bbox_area = sum([(bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) for bbox in bbox_list])

    overlap_percentage = (intersection_area / combined_bbox_area) * 100
    print("\n\n Overlap = " + str(overlap_percentage))

    # Display the original image with the bounding box
    pil_image.show()

    # # Save the image with bounding box if needed
    # pil_image_with_bbox = pil_image.copy()
    # draw = ImageDraw.Draw(pil_image_with_bbox)
    # draw.rectangle(bbox, outline="red", width=2)
    # pil_image_with_bbox.show()
    # pil_image_with_bbox.save("image_with_bbox_2.jpg")

    # mask.save("image_with_bbox_3.jpg")

    # # Close the images
    # pil_image.close()
    # pil_image_with_bbox.close()

    # Save the image with all bounding boxes if needed
    img_with_all_bboxes = draw_all_bboxes(pil_image, bbox_list)
    img_with_all_bboxes.show()
    img_with_all_bboxes.save("image_with_all_bboxes.jpg")

    return overlap_percentage

if __name__ == "__main__":
    # Example usage
    image_path = "path/to/your/image.jpg"
    bounding_box = (100, 50, 300, 200)

    # Check the overlap of the bounding box with the image data and display the result
    overlap_percentage = check_bbox_overlap(image_path, bounding_box)
    print(f"Overlap Percentage: {overlap_percentage}%")
