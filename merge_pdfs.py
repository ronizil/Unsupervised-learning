import config
print(config.FIGURE_DIR)
from pathlib import Path
import string
from pypdf import PdfReader, PdfWriter, PageObject
from io import BytesIO
from reportlab.pdfgen import canvas
from config import FIGURE_DIR, MERGED_FIGURE_PATH, GRID_COLS, GRID_ROWS, PAGE_WIDTH, PAGE_HEIGHT, FONT_SIZE, LABEL_OFFSET, CELL_W, CELL_H

PDF_FILES = [
    str(FIGURE_DIR / "fig1A.pdf"),
    str(FIGURE_DIR / "fig1B.pdf"),
    str(FIGURE_DIR / "fig1C.pdf"),
    str(FIGURE_DIR / "fig1D.pdf"),
    str(FIGURE_DIR / "fig1E.pdf"),
    str(FIGURE_DIR / "fig1F.pdf"),
]


def scale_to_fit(page, w, h):
    pw, ph = float(page.mediabox.width), float(page.mediabox.height)
    s = min(w / pw, h / ph)
    page.scale_by(s)
    return page


def merge_figures_to_grid():
    writer = PdfWriter()
    letters = string.ascii_uppercase
    panels = []
    for idx, path in enumerate(PDF_FILES):
        reader = PdfReader(path)
        pg = reader.pages[0]
        panels.append((scale_to_fit(pg, CELL_W, CELL_H), letters[idx]))

    for i in range(0, len(panels), GRID_COLS * GRID_ROWS):
        blank = PageObject.create_blank_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
        buf = BytesIO()
        c = canvas.Canvas(buf, pagesize=(PAGE_WIDTH, PAGE_HEIGHT))
        c.setFont("Helvetica-Bold", FONT_SIZE)
        for j in range(GRID_COLS * GRID_ROWS):
            if i + j >= len(panels):
                break
            row, col = divmod(j, GRID_COLS)
            x = col * CELL_W
            y = PAGE_HEIGHT - (row + 1) * CELL_H
            pg, letter = panels[i + j]
            blank.merge_translated_page(pg, x, y)
            c.drawString(x + LABEL_OFFSET, y + CELL_H - FONT_SIZE - LABEL_OFFSET, letter)
        c.save()
        buf.seek(0)
        lbl = PdfReader(buf).pages[0]
        blank.merge_page(lbl)
        writer.add_page(blank)

    with open(MERGED_FIGURE_PATH, 'wb') as f:
        writer.write(f)
    print(f"✅ Merged figure saved to {MERGED_FIGURE_PATH}")
