import fitz  # PyMuPDF
from PIL import Image
import io
import re
import os
import json

pdf_path = "training_manual.pdf"
output_folder = "Figures"
json_file = "figures.json"

# Create output folder if not exists
os.makedirs(output_folder, exist_ok=True)

# Dictionary to store figure metadata
figures_dict = {}

def clean_filename(text):
    """Cleans figure titles for filenames by replacing invalid characters."""
    text = text.replace("\u2014", "-")  # Replace em-dash with hyphen
    text = text.replace("—", "-")       # Replace any remaining dashes
    text = re.sub(r"[<>:\"/\\|?*\[\]]", "", text)  # Remove invalid characters
    text = text.strip().replace(" ", "_")[:80]     # Replace spaces & limit length
    return text

def clean_figure_title(text):
    """Extract 'Figure X' and remove trailing punctuation."""
    match = re.match(r"Figure\s*(\d+)\.?\s*(.*)", text)
    if match:
        figure_number = match.group(1)
        figure_name = match.group(2).strip().strip(".").replace("\u2014", "-")
        return figure_number, figure_name
    return None, text.strip().strip(".").replace("\u2014", "-")

doc = fitz.open(pdf_path)

for page_number in range(len(doc)):
    images = []
    
    # Extract text blocks from the page and sort
    text_blocks = doc[page_number].get_text("blocks")
    text_blocks = sorted(text_blocks, key=lambda b: b[1])  # sort by Y position

    # Extract images
    for img_index, img in enumerate(doc[page_number].get_images(full=True)):
        xref = img[0]
        base_image = doc.extract_image(xref)
        image_bytes = base_image["image"]
        image_ext = base_image["ext"]  # e.g. 'png' or 'jpeg'

        image_obj = Image.open(io.BytesIO(image_bytes))
        
        # Try to get the Y position
        img_rect = doc[page_number].get_image_rects(xref)
        y_pos = img_rect[0].y0 if img_rect else 0

        images.append((y_pos, image_obj, image_ext))

    if not images:
        print(f"No images found on page {page_number+1}, skipping...")
        continue

    # Sort images by Y position
    images.sort(key=lambda x: x[0])

    # If multiple images, merge vertically
    if len(images) > 1:
        total_height = sum(img[1].height for img in images)
        max_width = max(img[1].width for img in images)
        merged_image = Image.new("RGB", (max_width, total_height))

        y_offset = 0
        for _, img_obj, img_ext in images:
            merged_image.paste(img_obj, (0, y_offset))
            y_offset += img_obj.height
    else:
        merged_image = images[0][1]
        img_ext = images[0][2]

    # Find the closest text block containing "Figure"
    figure_title = None
    for block in text_blocks:
        block_text = block[4].strip()
        if block_text.startswith("Figure"):
            figure_title = block_text
            break

    if figure_title is None:
        print(f"No figure title found on page {page_number+1}, skipping image.")
        continue

    figure_number, cleaned_title = clean_figure_title(figure_title)
    sanitized_title = clean_filename(cleaned_title)

    if figure_number:
        figure_key = f"page_{page_number+1}_Figure_{figure_number}"
    else:
        figure_key = f"page_{page_number+1}"

    # Name the final file
    formatted_filename = f"{sanitized_title}.{img_ext}"
    filepath = os.path.join(output_folder, formatted_filename)
    merged_image.save(filepath)

    # Store BOTH the human-readable title and the filename in the JSON
    figures_dict[figure_key] = {
        "title": cleaned_title,        # e.g. "Engine Diagram"
        "filename": formatted_filename # e.g. "Engine_Diagram.png"
    }

    print(f"Saved merged image: {filepath}")

doc.close()

# Save metadata as JSON
with open(json_file, "w", encoding="utf-8") as f:
    json.dump(figures_dict, f, indent=4)

print(f"✅ Image extraction complete! Metadata saved to {json_file}.")
