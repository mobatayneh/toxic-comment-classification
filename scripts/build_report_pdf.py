"""Build the corrected report PDF from reviewed Markdown and aggregate figures.

No raw comments, model files, remote resources, or training are accessed.
Render the generated PDF with pdftoppm and inspect every page before delivery.
"""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image, KeepTogether, NextPageTemplate, PageBreak,
    PageTemplate, Paragraph, Spacer, Table, TableStyle,
)

BLUE = colors.HexColor("#245A81")
CHARCOAL = colors.HexColor("#313941")
GREY = colors.HexColor("#596572")
LIGHT_GREY = colors.HexColor("#EEF2F5")
MARGIN = 46
PORTRAIT_WIDTH = A4[0] - 2 * MARGIN
FIGURES = (
    ("../colab_efficient_v1/plots/training_losses.png", "Completed Colab training and validation losses", "Returned full-cohort random RoBERTa run: five epochs, with checkpoint selection by minimum validation loss at epoch 4. These validation observations informed selection; they are not internal-test measurements."),
    ("../colab_efficient_v1/plots/validation_per_label.png", "Colab validation selection diagnostics", "Average precision uses continuous scores; F1 is shown at 0.5 and at the six frozen validation-selected thresholds. Weak rare-label performance remains visible; these are selection diagnostics, not calibrated probabilities or production-readiness evidence."),
    ("../colab_efficient_v1/plots/validation_baseline_comparison.png", "Models on the same validation rows", "The saved CPU reference has higher aggregate AP and F1 than the returned random RoBERTa on the same 23,878 validation examples. Both used validation for selection; no independent neural test or statistical-significance claim is made."),
    ("historical_training_losses.png", "Historical loss trajectories", "Recorded notebook outputs, with the selected minimum-loss epochs marked. These are validation histories, not rerun measurements."),
    ("historical_per_label_report.png", "Historical precision, recall and F1", "FINAL_Report's retained final-epoch per-label output. The selected checkpoint and these prediction arrays were not synchronized in the original workflow."),
    ("real_source_label_prevalence.png", "Later supplied source cohort", "Positive counts refer to the supplied training-source rows before the seven normalized-empty comments are excluded. The six labels overlap."),
    ("baseline_internal_test_metrics.png", "New CPU exploratory internal test", "A distinct 70/15/15 grouped protocol on a historically examined cohort. Thresholds are selected on validation; this figure does not compare the CPU model with the historical neural model."),
)


def register_fonts() -> tuple[str, str, str]:
    """Use embedded readable fonts when available, with portable fallbacks."""
    font_dir = Path("/usr/share/fonts/truetype/dejavu")
    files = (
        ("ReviewSans", "DejaVuSans.ttf"),
        ("ReviewSans-Bold", "DejaVuSans-Bold.ttf"),
        ("ReviewMono", "DejaVuSansMono.ttf"),
    )
    if all((font_dir / filename).is_file() for _, filename in files):
        for name, filename in files:
            pdfmetrics.registerFont(TTFont(name, str(font_dir / filename)))
        pdfmetrics.registerFontFamily("ReviewSans", normal="ReviewSans", bold="ReviewSans-Bold", italic="ReviewSans", boldItalic="ReviewSans-Bold")
        return "ReviewSans", "ReviewSans-Bold", "ReviewMono"
    return "Helvetica", "Helvetica-Bold", "Courier"


def ascii_dashes(value: str) -> str:
    return value.translate(str.maketrans({"\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-", "\u2212": "-"}))


def inline_markup(value: str, mono_font: str) -> str:
    """Convert bounded Markdown inline styling to ReportLab paragraphs."""
    text = html.escape(ascii_dashes(value))

    def link(match: re.Match[str]) -> str:
        label, target = match.groups()
        if target.startswith(("https://", "http://")):
            return f'<link href="{target}" color="#245A81">{label}</link>'
        return label

    text = re.sub(r"\[([^\]]+)\]\(([^\s)]+)\)", link, text)
    text = re.sub(r"`([^`]+)`", lambda match: f'<font name="{mono_font}" size="8.4">{match.group(1)}</font>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*([^*]+)\*", r"<i>\1</i>", text)
    return text


def styles_for(font: str, bold_font: str) -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "body": ParagraphStyle("ReviewBody", parent=base["BodyText"], fontName=font, fontSize=9.6, leading=14, textColor=CHARCOAL, spaceAfter=7.2, alignment=TA_LEFT, splitLongWords=True),
        "title": ParagraphStyle("ReviewTitle", fontName=bold_font, fontSize=21, leading=27, textColor=BLUE, spaceAfter=17, keepWithNext=True),
        "h2": ParagraphStyle("ReviewH2", fontName=bold_font, fontSize=13, leading=18, textColor=BLUE, spaceBefore=15, spaceAfter=8, keepWithNext=True),
        "h3": ParagraphStyle("ReviewH3", fontName=bold_font, fontSize=10.4, leading=15, textColor=CHARCOAL, spaceBefore=10, spaceAfter=6, keepWithNext=True),
        "table": ParagraphStyle("ReviewTable", fontName=font, fontSize=8.1, leading=11.4, textColor=CHARCOAL, splitLongWords=True),
        "table_header": ParagraphStyle("ReviewTableHeader", fontName=bold_font, fontSize=8.1, leading=11.4, textColor=colors.white, splitLongWords=True),
        "note": ParagraphStyle("ReviewNote", fontName=font, fontSize=8.4, leading=12.5, textColor=GREY, spaceAfter=8),
        "status": ParagraphStyle("ReviewStatus", fontName=font, fontSize=9, leading=13.5, textColor=CHARCOAL, backColor=LIGHT_GREY, borderPadding=10, spaceBefore=2, spaceAfter=13),
    }


def table_from_markdown(lines: list[str], styles: dict[str, ParagraphStyle], mono_font: str) -> Table:
    raw = [[cell.strip() for cell in line.strip().strip("|").split("|")] for line in lines]
    rows = [raw[0], *raw[2:]]
    count = len(rows[0])
    if any(len(row) != count for row in rows):
        raise ValueError("Markdown table has inconsistent column counts")
    if count == 6 and rows[0][0] == "Notebook":
        weights = [1.9, 1.8, 0.6, 0.85, 0.85, 1.0]
    elif count == 5 and rows[0][0] == "Epoch":
        weights = [0.48, 1, 1, 1, 1]
    elif count == 4 and rows[0][0] == "Label":
        weights = [1.6, 1, 1, 1]
    else:
        weights = [max(9, min(30, max(len(row[column].replace("`", "")) for row in rows))) for column in range(count)]
    widths = [PORTRAIT_WIDTH * weight / sum(weights) for weight in weights]
    cells = [
        [Paragraph(inline_markup(value, mono_font), styles["table_header" if row_index == 0 else "table"]) for value in row]
        for row_index, row in enumerate(rows)
    ]
    table = Table(cells, colWidths=widths, repeatRows=1, hAlign="LEFT", spaceBefore=4, spaceAfter=10)
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), BLUE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LINEBELOW", (0, -1), (-1, -1), 0.5, colors.HexColor("#D5DDE3")),
    ]
    for row_index in range(1, len(cells)):
        if row_index % 2:
            commands.append(("BACKGROUND", (0, row_index), (-1, row_index), LIGHT_GREY))
    table.setStyle(TableStyle(commands))
    return table


def markdown_story(source: str, styles: dict[str, ParagraphStyle], mono_font: str) -> list:
    story: list = []
    lines = source.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line or line.startswith("<!--"):
            index += 1
            continue
        if line.startswith("|"):
            table_lines = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index])
                index += 1
            story.append(KeepTogether([table_from_markdown(table_lines, styles, mono_font)]))
            continue
        if line.startswith("### "):
            story.append(Paragraph(inline_markup(line[4:], mono_font), styles["h3"]))
        elif line.startswith("## "):
            story.append(Paragraph(inline_markup(line[3:], mono_font), styles["h2"]))
        elif line.startswith("# "):
            story.append(Paragraph(inline_markup(line[2:], mono_font), styles["title"]))
        elif line.startswith("- "):
            story.append(Paragraph(inline_markup(line[2:], mono_font), styles["body"], bulletText="-"))
        elif line.startswith("!["):
            # Figures have their own readable landscape appendix pages.
            pass
        else:
            paragraph = [line]
            while index + 1 < len(lines):
                next_line = lines[index + 1].strip()
                if not next_line or next_line.startswith(("#", "|", "- ", "<!--", "![")):
                    break
                index += 1
                paragraph.append(next_line)
            style = styles["status"] if line.startswith("**Review date:") or line.startswith("**Status:") else styles["body"]
            story.append(Paragraph(inline_markup(" ".join(paragraph), mono_font), style))
        index += 1
    return story


def build_pdf(source: Path, output: Path, figures: Path, include_figures: bool) -> None:
    font, bold_font, mono_font = register_fonts()
    styles = styles_for(font, bold_font)
    source_text = source.read_text(encoding="utf-8")
    review_date = re.search(r"\*\*Review date:\*\*\s*([^\.\n]+)", source_text)
    footer = "Evidence review" + (" | " + review_date.group(1).strip() if review_date else "")

    def decorate(canvas, document):
        width, height = canvas._pagesize
        canvas.saveState()
        canvas.setFont(font, 8)
        canvas.setFillColor(GREY)
        canvas.drawString(MARGIN, height - 27, "TOXIC COMMENT CLASSIFICATION / CORRECTED EVIDENCE REPORT")
        canvas.setStrokeColor(colors.HexColor("#D5DDE3"))
        canvas.setLineWidth(0.5)
        canvas.line(MARGIN, 34, width - MARGIN, 34)
        canvas.drawString(MARGIN, 21, footer)
        canvas.drawRightString(width - MARGIN, 21, str(document.page))
        canvas.restoreState()

    document = BaseDocTemplate(
        str(output), pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=MARGIN,
        title="Toxic Comment Classification: Corrected Evidence Report",
        author="Original academic report credit: Mohammad Albatayneh",
    )
    document.addPageTemplates([
        PageTemplate(id="Portrait", frames=[Frame(MARGIN, MARGIN, PORTRAIT_WIDTH, A4[1] - 2 * MARGIN, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)], onPage=decorate, pagesize=A4),
        PageTemplate(id="Landscape", frames=[Frame(MARGIN, MARGIN, landscape(A4)[0] - 2 * MARGIN, landscape(A4)[1] - 2 * MARGIN, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)], onPage=decorate, pagesize=landscape(A4)),
    ])
    story = markdown_story(source_text, styles, mono_font)
    if include_figures:
        for index, (filename, title, caption) in enumerate(FIGURES, start=1):
            path = figures / filename
            if not path.is_file():
                raise FileNotFoundError(f"Required report figure is missing: {path}")
            story.extend([NextPageTemplate("Landscape"), PageBreak()])
            story.append(Paragraph(f"Figure {index}. {title}", styles["h2"]))
            chart = Image(str(path))
            maximum_width = landscape(A4)[0] - 2 * MARGIN
            maximum_height = landscape(A4)[1] - 2 * MARGIN - 85
            ratio = min(maximum_width / chart.imageWidth, maximum_height / chart.imageHeight)
            chart.drawWidth = chart.imageWidth * ratio
            chart.drawHeight = chart.imageHeight * ratio
            chart.hAlign = "LEFT"
            story.append(chart)
            story.append(Spacer(1, 8))
            story.append(Paragraph(caption, styles["note"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    document.build(story)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=root / "reports/corrected_report.md")
    parser.add_argument("--output", type=Path, default=root / "reports/corrected_report.pdf")
    parser.add_argument("--figures", type=Path, default=root / "reports/figures")
    parser.add_argument("--no-figures", action="store_true")
    arguments = parser.parse_args()
    build_pdf(arguments.source, arguments.output, arguments.figures, not arguments.no_figures)
    print(arguments.output)


if __name__ == "__main__":
    main()
