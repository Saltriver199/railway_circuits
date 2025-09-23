import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Circle, Polygon, Rectangle
import numpy as np
import re
import os
from matplotlib.backends.backend_pdf import PdfPages
import math
import sys
from collections import OrderedDict

# === Function to merge ranges ===
def merge_ranges(ranges, merge_adjacent=True):
    """
    Merge overlapping (and optionally adjacent) integer index ranges.
    ranges: list of (start, end) tuples where start<=end
    merge_adjacent: if True, ranges like (0,5) and (6,11) will merge into (0,11).
                    if False, adjacent ranges remain separate.
    """
    if not ranges:
        return []
    sorted_ranges = sorted(ranges)
    merged = [list(sorted_ranges[0])]
    for current in sorted_ranges[1:]:
        last = merged[-1]
        if merge_adjacent:
            cond = (current[0] <= last[1] + 1)
        else:
            cond = (current[0] <= last[1])  # only merge if overlapping, not merely adjacent
        if cond:
            last[1] = max(last[1], current[1])
        else:
            merged.append(list(current))
    return [tuple(r) for r in merged]

# === Function to get circuit name ===
def get_block_circuit_name(df_block):
    if 'row' in df_block.columns:
        s = (
            df_block['row']
            .dropna().astype(str).str.strip()
            .replace('', pd.NA).dropna()
        )
        if not s.empty:
            return s.iloc[0]
    return ""

# === Function to draw circuit name with circle ===
def draw_circuit_name(ax, x, y, circuit_name, x_offset=0.65):
    circle_center = (x + x_offset, y)
    ax.add_patch(Circle(circle_center, radius=0.22,
                        edgecolor='black', facecolor='white', linewidth=0.8))
    ax.text(x + x_offset, y, circuit_name, ha='center', va='center',
            fontsize=18, fontweight='bold')

# === Function to draw junction box with big text ===
def draw_junction_box(ax, x, y, junction_name, rect_pad=0.2):
    text_raw = str(junction_name).strip()
    if not text_raw:
        return
    s = text_raw
    text_width = len(s) * 0.35
    text_height = 1.0
    font_size = max(20, min(35, 35 - (len(s) - 5) * 0.5))
    rect_width = text_width + rect_pad * 10
    rect_x = x - rect_width / 2
    rect_y = y - text_height / 2
    rect = Rectangle((rect_x, rect_y), rect_width, text_height,
                     linewidth=2, edgecolor='black', facecolor='none')
    ax.add_patch(rect)
    ax.text(x, y, s, ha='center', va='center',
            fontsize=font_size, fontweight='bold', zorder=5)

# === Relay input symbol ===
def draw_relay_input(ax, x_left, x_right, y=0, scale=1.0, text='RELAY', anchor_to_v_tip=False, v_offset=-0.5):
    if x_left is None or x_right is None:
        return
    if x_left > x_right:
        x_left, x_right = x_right, x_left
    span = max(1e-6, float(x_right) - float(x_left))
    center = (x_left + x_right) / 2.0
    tri_base = min(max(span * 0.18, 0.25 * scale), span * 0.45)
    tri_height = tri_base * 0.25 * scale
    v_depth = tri_height * 0.9
    left_notch = (center - tri_base / 2.0, y - v_offset)
    right_notch = (center + tri_base / 2.0, y - v_offset)
    notch_top = (center, y + tri_height - v_offset)
    v_tip = (center, y - v_depth - v_offset)
    pad = min(span * 0.02 + 0.02 * scale, span * 0.05)
    left_wire_x_start = x_left - pad
    right_wire_x_end = x_right + pad
    slant_size = min(0.18 * scale, tri_base * 0.35)
    end_slant_size = slant_size * 0.6
    ax.plot([left_wire_x_start - end_slant_size, left_wire_x_start],
            [y - end_slant_size - v_offset, y - v_offset], color='black', linewidth=1.4)
    ax.plot([right_wire_x_end, right_wire_x_end + end_slant_size],
            [y - v_offset, y - end_slant_size - v_offset], color='black', linewidth=1.4)
    if anchor_to_v_tip:
        ax.plot([left_notch[0], v_tip[0]], [left_notch[1], v_tip[1]], linewidth=1.4, color='black')
        ax.plot([right_notch[0], v_tip[0]], [right_notch[1], v_tip[1]], linewidth=1.4, color='black')
    else:
        ax.plot([left_wire_x_start, left_notch[0]], [y - v_offset, left_notch[1]], linewidth=1.4, color='black')
        ax.plot([right_notch[0], right_wire_x_end], [right_notch[1], y - v_offset], linewidth=1.4, color='black')
    ax.plot([left_notch[0], notch_top[0]], [left_notch[1], notch_top[1]], color='black', linewidth=1.4)
    ax.plot([right_notch[0], notch_top[0]], [right_notch[1], notch_top[1]], color='black', linewidth=1.4)
    text_y_offset = 0.15 * scale
    text_y = notch_top[1] + text_y_offset
    ax.text(center, text_y, str(text), ha='center', va='bottom',
            fontsize=int(18 * scale), fontweight='bold')

    # Add two small vertical lines on upper side if text is "LIGHT"
    if str(text).strip().upper() == "LIGHT":
        line_length = 0.3 * scale
        gap = 0.25 * scale  # increase gap between lines
        left_line_x = center - gap
        right_line_x = center + gap
        base_y = y - v_offset - 0.4 * scale  # move lines slightly down side 

        # Left vertical line
        ax.plot([left_line_x, left_line_x], [base_y, base_y + line_length], color='black', linewidth=1.4)
        # Right vertical line
        ax.plot([right_line_x, right_line_x], [base_y, base_y + line_length], color='black', linewidth=1.4)


# === Relay output symbol ===
def draw_relay_output(ax, x_left, x_right, y=0, scale=1.0, text='RELAY', anchor_to_v_tip=False, v_offset=0.5):
    if x_left is None or x_right is None:
        return
    if x_left > x_right:
        x_left, x_right = x_right, x_left
    span = max(1e-6, float(x_right) - float(x_left))
    center = (x_left + x_right) / 2.0
    tri_base = min(max(span * 0.18, 0.25 * scale), span * 0.45)
    tri_height = tri_base * 0.25 * scale
    v_depth = tri_height * 0.9
    left_notch = (center - tri_base / 2.0, y - v_offset)
    right_notch = (center + tri_base / 2.0, y - v_offset)
    notch_bottom = (center, y - tri_height - v_offset)
    v_tip = (center, y + v_depth - v_offset)
    pad = min(span * 0.02 + 0.02 * scale, span * 0.05)
    left_wire_x_start = x_left - pad
    right_wire_x_end = x_right + pad
    slant_size = min(0.18 * scale, tri_base * 0.35)
    end_slant_size = slant_size * 0.6
    ax.plot([left_wire_x_start - end_slant_size, left_wire_x_start],
            [y + end_slant_size - v_offset, y - v_offset], color='black', linewidth=1.4)
    ax.plot([right_wire_x_end, right_wire_x_end + end_slant_size],
            [y - v_offset, y + end_slant_size - v_offset], color='black', linewidth=1.4)
    if anchor_to_v_tip:
        ax.plot([left_notch[0], v_tip[0]], [left_notch[1], v_tip[1]], linewidth=1.4, color='black')
        ax.plot([right_notch[0], v_tip[0]], [right_notch[1], v_tip[1]], linewidth=1.4, color='black')
    else:
        ax.plot([left_wire_x_start, left_notch[0]], [y - v_offset, left_notch[1]], linewidth=1.4, color='black')
        ax.plot([right_notch[0], right_wire_x_end], [right_notch[1], y - v_offset], linewidth=1.4, color='black')
    ax.plot([left_notch[0], notch_bottom[0]], [left_notch[1], notch_bottom[1]], color='black', linewidth=1.4)
    ax.plot([right_notch[0], notch_bottom[0]], [right_notch[1], notch_bottom[1]], color='black', linewidth=1.4)
    text_y_offset = 0.15 * scale
    text_y = notch_bottom[1] - text_y_offset
    display_text = str(text)
    if len(display_text) >= 8:
        words = display_text.split()
        if len(words) > 1:
            line1 = " ".join(words[:-1])
            line2 = words[-1]
            display_text = f"{line1}\n{line2}"
        else:
            display_text = display_text[:8] + "\n" + display_text[8:]
    ax.text(center, text_y, display_text,
            ha='center', va='top',
            fontsize=int(18 * scale),
            fontweight='bold', linespacing=1.2)
    
    # Add two small vertical lines on upper side if text is "LIGHT"
    # Add two small vertical lines on upper side if text is "LIGHT"
    if str(text).strip().upper() == "LIGHT":
        line_length = 0.3 * scale
        gap = 0.25 * scale  # increase gap between lines
        left_line_x = center - gap
        right_line_x = center + gap
        base_y = y - v_offset + 0.1 * scale  # move lines slightly upward

        # Left vertical line
        ax.plot([left_line_x, left_line_x], [base_y, base_y + line_length], color='black', linewidth=1.4)
        # Right vertical line
        ax.plot([right_line_x, right_line_x], [base_y, base_y + line_length], color='black', linewidth=1.4)


def draw_group_top_symbol(
    ax,
    x_start,
    x_end,
    y,
    texts='R1',
    scale=1.0,
    input_connected='N',
    spacing=0.3,                # Controls vertical spacing between stacked symbols
    x_offset=0.3,               # Horizontal offset for duplicate symbols
    diagonal_length=0.21,       # Length of the diagonal bar
    split_text=True,            # Whether to split text longer than 3 characters
    split_length=3,             # Length threshold for splitting text
    draw_diagonal=True,         # Whether to draw the diagonal bar (/)
    draw_vertical=True,         # Whether to draw the small vertical line (|)
    vertical_linewidth=1.2,     # Line width for the small vertical line
    diagonal_linewidth=1.2      # Line width for the diagonal bar
):
    if isinstance(texts, str):
        texts = [texts]
    x = (x_start + x_end) / 2.0 if abs(x_end - x_start) < 0.1 else x_start
    line_extension = 0.35 * scale
    relay_gap = 0.1 * scale if str(input_connected).strip().upper() == 'Y' else 0.0
    base_y = y + relay_gap

    num_texts = len(texts)
    spacing = spacing * scale  # Spacing for stacking symbols
    total_extra = (num_texts - 1) * spacing
    line_extension += total_extra

    # Vertical center line (going up)
    ax.plot([x, x], [base_y, base_y + line_extension], color='black', linewidth=1)

    # One \______ style segment at the top (diagonal first, horizontal after)
    total_width = x_end - x_start
    horizontal_ratio = 0.85
    y_top = base_y + 0.11 * scale
    drop_height = 0.08 * scale
    y_bottom = y_top - drop_height

    seg_x0 = x_start
    seg_x2 = x_end
    seg_x1 = seg_x0 + total_width * (1 - horizontal_ratio)  # end of diagonal

    # Diagonal down (\ shape) first
    ax.plot([seg_x1, seg_x0], [y_top, y_bottom], color='black', linewidth=1)

    # Horizontal line after diagonal (slightly lower)
    horizontal_offset = 0.08 * scale
    ax.plot([seg_x1, seg_x2], [y_bottom + horizontal_offset, y_bottom + horizontal_offset],
            color='black', linewidth=1)

    # Stack diagonal bars with small vertical/diagonal connectors if multiple texts
    diagonal_length = diagonal_length * scale
    y_shift = -0.17 * scale
    diag_offset = -0.05 * scale
    left_adjust = -0.04 * scale
    right_adjust = 0.04 * scale
    down_shift = -0.01 * scale
    prev_center_y = None
    x_offset = x_offset * scale  # Shift for duplicates

    for i in range(num_texts):
        extra_shift = i * spacing
        # Apply x_offset for duplicates (i > 0)
        current_x = x + x_offset if i > 0 else x
        left_y = base_y + line_extension - diagonal_length - y_shift + left_adjust + diag_offset + down_shift + extra_shift
        right_y = base_y + line_extension - y_shift + right_adjust + diag_offset + down_shift + extra_shift

        # --- Original diagonal bar (/) ---
        if draw_diagonal:
            ax.plot([current_x - diagonal_length / 2, current_x + diagonal_length / 2],
                    [left_y, right_y],
                    color='black', linewidth=diagonal_linewidth)

            # --- NEW: stacked diagonal connector between symbols ---
            if prev_center_y is not None:
                ax.plot([current_x - diagonal_length / 2, current_x + diagonal_length / 2],
                        [prev_center_y, (left_y + right_y) / 2.0],
                        color='black', linewidth=10)

        center_y = (left_y + right_y) / 2.0

        # --- Vertical connector (|) ---
        if draw_vertical and prev_center_y is not None:
            ax.plot([current_x, current_x], [prev_center_y, center_y],
                    color='black', linewidth=10)

        prev_center_y = center_y

        # --- Text label ---
        text_offset = -0.1 * scale
        text_y = base_y + line_extension + 0.1 - y_shift + text_offset + extra_shift
        display_text = str(texts[i]).strip()

        # Split text into two lines if enabled and longer than split_length
        if split_text and len(display_text) > split_length:
            mid = len(display_text) // 2
            display_text = display_text[:mid] + '\n' + display_text[mid:]

        ax.text(current_x, text_y, display_text, ha='center', va='bottom',
                fontsize=int(17 * scale), fontweight='bold')


def draw_group_bottom_symbol(ax, x_start, x_end, y, text='R1', scale=1.0, output_connected='N', choke_output_terminal=None):
    x = (x_start + x_end) / 2.0 if abs(x_end - x_start) < 0.1 else x_start
    line_extension = 0.35 * scale
    relay_gap = 0.1 * scale if str(output_connected).strip().upper() == 'Y' else 0.0
    base_y = y - relay_gap

    # Vertical center line
    if choke_output_terminal is None:
        ax.plot([x, x], [base_y, base_y - line_extension], color='black', linewidth=1)

    # One /‾‾‾‾‾‾ style segment at the bottom (diagonal first, horizontal after)
    total_width = x_end - x_start
    horizontal_ratio = 0.85
    y_bottom = base_y - 0.11 * scale
    rise_height = 0.08 * scale
    y_top = y_bottom + rise_height

    seg_x0 = x_start
    seg_x2 = x_end
    seg_x1 = seg_x0 + total_width * (1 - horizontal_ratio)

    # Diagonal up (/ shape) first
    ax.plot([seg_x1, seg_x0], [y_bottom, y_top], color='black', linewidth=1)

    # Horizontal line after diagonal (slightly lower)
    horizontal_offset = 0.08 * scale
    ax.plot([seg_x1, seg_x2], [y_top - horizontal_offset, y_top - horizontal_offset],
            color='black', linewidth=1)

    # Diagonal bar
    diagonal_length = 0.21 * scale
    y_shift = 0.07 * scale
    diag_offset = 0.05 * scale
    left_adjust = 0.04 * scale
    right_adjust = -0.04 * scale
    diagonal_down_shift = 0.02 * scale
    left_y = base_y - line_extension - diagonal_length + y_shift + left_adjust + diag_offset - diagonal_down_shift
    right_y = base_y - line_extension + y_shift + right_adjust + diag_offset - diagonal_down_shift

    if choke_output_terminal is None:
        ax.plot([x - diagonal_length/2, x + diagonal_length/2],
                [left_y, right_y],
                color='black', linewidth=1.2)

    # If choke output terminal is specified, add horizontal line and terminal label
    if choke_output_terminal is not None:


        # Diagonal bar
        diagonal_length = 0.21 * scale
        y_shift = 0.07 * scale
        diag_offset = 0.05 * scale
        left_adjust = 0.04 * scale
        right_adjust = -0.04 * scale
        diagonal_down_shift = 0.02 * scale
        # Recalculate left_y and right_y
        left_y = base_y - line_extension - diagonal_length + y_shift + left_adjust + diag_offset - diagonal_down_shift
        right_y = base_y - line_extension + y_shift + right_adjust + diag_offset - diagonal_down_shift
        # Draw the diagonal line
        dx = 0.5  # adjust this value as needed
        ax.plot([x - diagonal_length/2 + dx, x + diagonal_length/2 + dx],
                [left_y, right_y],
                color='black', linewidth=1.2)


        # Horizontal line extending to the right
        horiz_length = 0.5 * scale
        left_shift = 0.1  # how much to move left
        end_x = x + diagonal_length/2 + horiz_length - left_shift  # move end slightly left
        start_x = x + diagonal_length/2 - left_shift  # move start slightly left
        line_y_shift = 2.78   # move horizontal line upward
        line_y = right_y + line_y_shift
        # Horizontal line
        ax.plot([start_x, end_x], [line_y, line_y], color='black', linewidth=1.2)

        # Vertical line going downward from end of horizontal line
        vertical_length = 2.84  # length downward
        ax.plot([end_x, end_x], [line_y, line_y - vertical_length], color='black', linewidth=1.2)


    # Text label
    # Text label
    text_offset = 0.2 * scale

    # Adjust text_y and text_x based on whether choke_output_terminal is present
    if choke_output_terminal is not None:
        # Position text below the choke's horizontal line
        text_y = right_y - text_offset - 0.05  # Additional downward shift for choke
        text_x = x + 0.55  # move slightly to the right
    else:
        # Original position for non-choke cases
        text_y = base_y - line_extension - diagonal_length - 0.1 + y_shift + text_offset - 0.05
        text_x = x  # keep centered

    display_text = str(text).strip()

    # Split text into two lines if longer than 3 characters
    if len(display_text) > 3:
        mid = len(display_text) // 2
        display_text = display_text[:mid] + '\n' + display_text[mid:]

    # Draw text
    ax.text(text_x, text_y, display_text, ha='center', va='top',
            fontsize=int(17 * scale), fontweight='bold', linespacing=1.2)


# === Load Excel file path from command line (or prompt) ===
if len(sys.argv) > 1:
    EXCEL_FILE = sys.argv[1]
else:
    EXCEL_FILE = input("Enter Excel file path (e.g. C:\\Diagram\\RAILWAYPROJECT.xlsx) or press Enter to exit: ").strip()
    if not EXCEL_FILE:
        print("No Excel file provided. Exiting.")
        sys.exit(1)

if not os.path.exists(EXCEL_FILE):
    print(f"Error: Excel file not found at: {EXCEL_FILE}")
    sys.exit(1)

# Validate required sheets exist
try:
    xls = pd.ExcelFile(EXCEL_FILE)
except Exception as e:
    print(f"Unable to open Excel file: {e}")
    sys.exit(1)

required_sheets = ['terminal', 'junction_box', 'terminal_header', 'group', 'circuit']
available_sheets = [s.strip() for s in xls.sheet_names]
missing = [s for s in required_sheets if s not in available_sheets]
if missing:
    print(f"Excel file is missing required sheets: {missing}")
    print(f"Available sheets: {available_sheets}")
    sys.exit(1)

# Load StationDrawing for footer if available
df_title = None
try:
    df_title = pd.read_excel(EXCEL_FILE, sheet_name='StationDrawing')
    df_title.columns = df_title.columns.str.strip()
    print("Loaded StationDrawing sheet for footer.")
except Exception as e:
    print(f"Warning: Could not load StationDrawing sheet for footer: {e}. Footer will be skipped.")

# === Load Excel sheets ===
try:
    df = pd.read_excel(EXCEL_FILE, sheet_name='terminal')
    df.columns = df.columns.str.strip()
    df_junction = pd.read_excel(EXCEL_FILE, sheet_name='junction_box')
    df_junction.columns = df_junction.columns.str.strip()
    df_header = pd.read_excel(EXCEL_FILE, sheet_name='terminal_header')
    df_header.columns = df_header.columns.str.strip()
    df_group = pd.read_excel(EXCEL_FILE, sheet_name='group')
    df_group.columns = df_group.columns.str.strip()
    df_circuit = pd.read_excel(EXCEL_FILE, sheet_name='circuit')
    df_circuit.columns = df_circuit.columns.str.strip()
    df_choke = pd.read_excel(EXCEL_FILE, sheet_name='choketable')
    df_choke.columns = df_choke.columns.str.strip()
    df_resistor = pd.read_excel(EXCEL_FILE, sheet_name='resistortable')
    df_resistor.columns = df_resistor.columns.str.strip()
except Exception as e:
    print(f"Error reading required sheets from Excel file: {e}")
    sys.exit(1)
finally:
    try:
        xls.close()
    except Exception:
        pass

if 'spare' in df.columns:
    df.loc[df['spare'].astype(str).str.upper() == 'Y', 'input_left'] = 'SP'

# === STANDARDIZED DIMENSIONS ===
SYMBOL_HEIGHT = 0.6
SYMBOL_WIDTH = 0.35
SYMBOL_RADIUS = 0.15
CAPSULE_Y_CENTER_BASE = 3.6
y_top_bus_offset = 1.3
y_bottom_bus_offset = -1.3
stub_length = 0.74
CIRCUIT_GAP = 2.0
vertical_gap = 6.5

# Footer dimensions (adjusted to match row spacing)
footer_height = 2.75  # Adjusted footer height
footer_inch_add = 4.0  # Reduced additional inches for footer

# === Updated draw_header ===
def draw_header(ax, circuit_id, header_type, x_start, x_end, text, min_symbol_bottom=None,
                first_hook_x=None, last_hook_x=None, y_top_bus_group=0, y_bottom_bus_group=0, special_ha=False):
    top_y_offset = 0.1
    bottom_y_offset = 0.85
    if pd.isna(text) or str(text).strip() == '':
        return
    text = str(text).strip()
    if str(header_type).strip().upper() == 'WIREFROM':
        x_pos = first_hook_x if first_hook_x is not None else x_start - 0.05
        y_pos = y_top_bus_group + top_y_offset
        ax.text(x_pos, y_pos, text, ha='left', va='bottom', fontsize=17, fontweight='bold')
    elif str(header_type).strip().upper() == 'WIRETO':
        ha = 'center'
        x_pos = last_hook_x if last_hook_x is not None else (x_start + x_end) / 2.0
        if last_hook_x is not None:
            ha = 'left' if special_ha else 'right'
        min_symbol_bottom = y_bottom_bus_group - 0.2 if min_symbol_bottom is None else min_symbol_bottom
        text_offset = -0.15
        y_pos = min_symbol_bottom - bottom_y_offset + text_offset
        ax.text(x_pos, y_pos, text, ha=ha, va='top', fontsize=17, fontweight='bold')

# === Helper: Find row by terminal number ===
def find_row_by_term(term):
    if pd.isna(term):
        return None
    s = str(term).strip()
    if s.endswith('.0'):
        s = s[:-2]
    col = df['terminal_name'].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
    matches = df[col == s]
    return matches.iloc[0] if not matches.empty else None

# === Symbol Drawers ===
def draw_capsule(ax, x, y_center, terminal_name, input_left, input_right, output_left, output_right,
                 input_connected, output_connected):
    capsule_bottom = y_center - SYMBOL_HEIGHT / 2
    capsule_top = capsule_bottom + SYMBOL_HEIGHT
    top_circle_radius = SYMBOL_RADIUS * 0.8
    ax.add_patch(Circle((x, capsule_top), radius=top_circle_radius,
                        edgecolor='black', facecolor='white', linewidth=1))
    bottom_circle_radius = SYMBOL_RADIUS * 0.8
    ax.add_patch(Circle((x, capsule_bottom), radius=bottom_circle_radius,
                        edgecolor='black', facecolor='white', linewidth=1))
    line_offset = SYMBOL_WIDTH / 2
    extend = 0.11
    shift_left = 0.055
    shift_right = 0.055
    ax.plot([x - line_offset + shift_left, x - line_offset + shift_left],
            [capsule_bottom + bottom_circle_radius - extend, capsule_top - SYMBOL_RADIUS + extend],
            color='black', linewidth=1)
    ax.plot([x + line_offset - shift_right, x + line_offset - shift_right],
            [capsule_bottom + bottom_circle_radius - extend, capsule_top - SYMBOL_RADIUS + extend],
            color='black', linewidth=1)
    if pd.notna(terminal_name) and str(terminal_name).strip() != '':
        term_str = str(terminal_name)
        if term_str.endswith('.0'):
            term_str = term_str[:-2]
        ax.text(x, y_center, term_str, fontsize=12, ha='center', va='center')
    def format_text(t):
        t = str(t)
        if len(t) >= 8:
            words = t.split()
            if len(words) > 1:
                line1 = " ".join(words[:-1])
                line2 = words[-1]
                return f"{line1}\n{line2}"
            else:
                return t[:8] + "\n" + t[8:]
        return t
    input_left_offset = 0.005
    if pd.notna(input_left) and str(input_left).strip() != "":
        ax.text(x - input_left_offset, capsule_top + 0.18, format_text(input_left),
                fontsize=12, ha='right', va='bottom', rotation=90, linespacing=1.2)
    input_right_offset = 0.05
    if pd.notna(input_right) and str(input_right).strip() != "":
        ax.text(x + input_right_offset, capsule_top + 0.18, format_text(input_right),
                fontsize=12, ha='left', va='bottom', rotation=90, linespacing=1.2)
    output_left_offset = 0.005
    if pd.notna(output_left) and str(output_left).strip() != "":
        ax.text(x - output_left_offset, capsule_bottom - 0.15, format_text(output_left),
                fontsize=12, ha='right', va='top', rotation=90, linespacing=1.2)
    output_right_offset = 0.05
    if pd.notna(output_right) and str(output_right).strip() != "":
        ax.text(x + output_right_offset, capsule_bottom - 0.18, format_text(output_right),
                fontsize=12, ha='left', va='top', rotation=90, linespacing=1.2)
    top_conn = (x, capsule_top + SYMBOL_RADIUS)
    bottom_conn = (x, capsule_bottom - bottom_circle_radius)
    ic = 'Y' if str(input_connected).strip().upper() == 'Y' else 'N'
    oc = 'Y' if str(output_connected).strip().upper() == 'Y' else 'N'
    return top_conn, bottom_conn, ic, oc

def draw_s_fuse(ax, x, y_center, terminal_name,
                input_left=None, input_right=None,
                output_left=None, output_right=None,
                input_connected='N', output_connected='N'):
    fuse_top = y_center + SYMBOL_HEIGHT / 2
    fuse_bottom = y_center - SYMBOL_HEIGHT / 2
    top_circle_radius = SYMBOL_RADIUS * 0.8
    bottom_circle_radius = SYMBOL_RADIUS * 0.8

    # Draw circles
    ax.add_patch(Circle((x, fuse_top), top_circle_radius,
                        edgecolor='black', facecolor='white', linewidth=1))
    ax.add_patch(Circle((x, fuse_bottom), bottom_circle_radius,
                        edgecolor='black', facecolor='white', linewidth=1))

    # Curved middle connection
    start = (x, fuse_top - top_circle_radius)
    end = (x, fuse_bottom + bottom_circle_radius)
    ctrl1 = (x + SYMBOL_RADIUS * 2.2, y_center + SYMBOL_HEIGHT * 0.15)
    ctrl2 = (x - SYMBOL_RADIUS * 2.2, y_center - SYMBOL_HEIGHT * 0.15)
    t = np.linspace(0, 1, 100)
    xs = (1 - t)**3 * start[0] + 3 * (1 - t)**2 * t * ctrl1[0] + \
         3 * (1 - t) * t**2 * ctrl2[0] + t**3 * end[0]
    ys = (1 - t)**3 * start[1] + 3 * (1 - t)**2 * t * ctrl1[1] + \
         3 * (1 - t) * t**2 * ctrl2[1] + t**3 * end[1]
    ax.plot(xs, ys, color='black', linewidth=1, solid_capstyle='round')

    # Format function (same as capsule)
    def format_text(t):
        t = str(t)
        if len(t) >= 8:
            words = t.split()
            if len(words) > 1:
                line1 = " ".join(words[:-1])
                line2 = words[-1]
                return f"{line1}\n{line2}"
            else:
                return t[:8] + "\n" + t[8:]
        return t

    # Terminal name
    if pd.notna(terminal_name) and str(terminal_name).strip() != '':
        term_str = str(terminal_name)
        if term_str.endswith('.0'):
            term_str = term_str[:-2]
        ax.text(x - 0.1, y_center + 0.01, term_str,
                ha='center', va='center', fontsize=12)

    # Input/Output labels with auto split
    input_left_offset = 0.005
    if pd.notna(input_left) and str(input_left).strip() != "":
        ax.text(x - input_left_offset, fuse_top + 0.18,
                format_text(input_left), fontsize=12,
                ha='right', va='bottom', rotation=90, linespacing=1.2)

    input_right_offset = 0.05
    if pd.notna(input_right) and str(input_right).strip() != "":
        ax.text(x + input_right_offset, fuse_top + 0.18,
                format_text(input_right), fontsize=12,
                ha='left', va='bottom', rotation=90, linespacing=1.2)

    output_left_offset = 0.005
    if pd.notna(output_left) and str(output_left).strip() != "":
        ax.text(x - output_left_offset, fuse_bottom - 0.15,
                format_text(output_left), fontsize=12,
                ha='right', va='top', rotation=90, linespacing=1.2)

    output_right_offset = 0.05
    if pd.notna(output_right) and str(output_right).strip() != "":
        ax.text(x + output_right_offset, fuse_bottom - 0.18,
                format_text(output_right), fontsize=12,
                ha='left', va='top', rotation=90, linespacing=1.2)

    # Connections
    top_conn = (x, fuse_top + top_circle_radius)
    bottom_conn = (x, fuse_bottom - bottom_circle_radius)
    ic = 'Y' if str(input_connected).strip().upper() == 'Y' else 'N'
    oc = 'Y' if str(output_connected).strip().upper() == 'Y' else 'N'
    return top_conn, bottom_conn, ic, oc


def draw_choke(ax, x, y_center, terminal_name):
    row = find_row_by_term(terminal_name)
    input_left = input_right = output_left = output_right = None
    input_connected = 'N'
    output_connected = 'N'
    if row is not None:
        input_left = row.get('input_left')
        input_right = row.get('input_right')
        output_left = row.get('output_left')
        output_right = row.get('output_right')
        input_connected = row.get('input_connected', 'N')
        output_connected = row.get('output_connected', 'N')
    choke_top = y_center + SYMBOL_HEIGHT / 2
    choke_bottom = y_center - SYMBOL_HEIGHT / 2
    top_center = np.array([x, choke_top])
    bottom_center = np.array([x, choke_bottom])
    for center in [top_center, bottom_center]:
        circ = Circle(center, SYMBOL_RADIUS, edgecolor='black', facecolor='white', linewidth=1)
        ax.add_patch(circ)
    start_y = choke_top - SYMBOL_RADIUS
    end_y = choke_bottom + SYMBOL_RADIUS
    num_coils = 4
    t = np.linspace(0, num_coils * np.pi, 100)
    xs = x + SYMBOL_RADIUS * 1.5 * np.sin(t)
    ys = start_y - (start_y - end_y) * (t / (num_coils * np.pi))
    ax.plot(xs, ys, color='black', linewidth=1, solid_capstyle='round')
    if pd.notna(terminal_name) and str(terminal_name).strip() != '':
        term_str = str(terminal_name)
        if term_str.endswith('.0'):
            term_str = term_str[:-2]
        ax.text(x - 0.3, y_center + 0.1, term_str, ha='center', va='center', fontsize=10.5, fontweight='bold')
    label_offset = 0.2
    if pd.notna(input_left) and str(input_left).strip() != "":
        ax.text(x - label_offset, choke_top + 0.15, str(input_left), fontsize=6, ha='right', va='bottom', rotation=90)
    if pd.notna(input_right) and str(input_right).strip() != "":
        ax.text(x + label_offset, choke_top + 0.15, str(input_right), fontsize=6, ha='left', va='bottom', rotation=90)
    if pd.notna(output_left) and str(output_left).strip() != "":
        ax.text(x - label_offset, choke_bottom - 0.15, str(output_left), fontsize=6, ha='right', va='top', rotation=90)
    if pd.notna(output_right) and str(output_right).strip() != "":
        ax.text(x + label_offset, choke_bottom - 0.15, str(output_right), fontsize=6, ha='left', va='top', rotation=90)
    top_conn = (x, choke_top + SYMBOL_RADIUS)
    bottom_conn = (x, choke_bottom - SYMBOL_RADIUS)
    ic = 'Y' if str(input_connected).strip().upper() == 'Y' else 'N'
    oc = 'Y' if str(output_connected).strip().upper() == 'Y' else 'N'
    return top_conn, bottom_conn, ic, oc

# === Updated draw_horizontal_choke ===
def draw_horizontal_choke(ax, x_center, y_center, label='CHOKE',
                          box_width=0.45, box_height=0.35,
                          special_end=False, output_label=''):
    # shift downward
    y_shift = -0.5   # adjust this value to move more/less down

    left_x = x_center - box_width / 2
    right_x = x_center + box_width / 2
    bottom_y = (y_center + y_shift) - box_height / 2

    # Draw rounded box
    choke_box = FancyBboxPatch((left_x, bottom_y),
                               box_width, box_height,
                               boxstyle="round,pad=0.02",
                               edgecolor='black', facecolor='white', linewidth=1.5)
    ax.add_patch(choke_box)

    # Draw label (moved with same y_shift)
    ax.text(x_center, y_center + y_shift, label,
            fontsize=18, ha='center', va='center', fontweight='bold')

    # Connection lines
    line_length = 0.075
    delta = 0.02
    vert_line_height = 0.5  # how tall the vertical lines go upward

    # Left connection line 
    left_horiz_start = left_x - line_length - delta
    left_horiz_end   = left_x - delta

    ax.plot([left_horiz_start, left_horiz_end],
            [y_center + y_shift, y_center + y_shift],
            color='black', linewidth=1)

    # vertical up from left end (moved slightly left)
    v_offset = 0.005  # shift amount
    ax.plot([left_horiz_start - v_offset, left_horiz_start - v_offset],
            [y_center + y_shift, y_center + y_shift + vert_line_height],
            color='black', linewidth=1)

    vert_x = None
    if not special_end:
        # Right connection line
        right_horiz_start = right_x + delta
        right_horiz_end   = right_x + line_length + delta
        ax.plot([right_horiz_start, right_horiz_end],
                [y_center + y_shift, y_center + y_shift],
                color='black', linewidth=1)

        # vertical up from right end (moved slightly right)
        ax.plot([right_horiz_end + v_offset, right_horiz_end + v_offset],
                [y_center + y_shift, y_center + y_shift + vert_line_height],
                color='black', linewidth=1)

    else:
        # special right end
        horiz_length = 0.2
        slant_size = 0.3
        vertical_length = 0.5
        ax.plot([right_x + delta, right_x + delta + horiz_length],
                [y_center + y_shift, y_center + y_shift],
                color='black', linewidth=1.4)
        end_horiz_x = right_x + delta + horiz_length
        ax.plot([end_horiz_x, end_horiz_x + slant_size],
                [y_center + y_shift, y_center + y_shift - slant_size],
                color='black', linewidth=1.4)
        vert_x = end_horiz_x + slant_size
        vert_top_y = y_center + y_shift - slant_size
        vert_bottom_y = vert_top_y - vertical_length
        ax.plot([vert_x, vert_x], [vert_top_y, vert_bottom_y],
                color='black', linewidth=1.4)
        # label
        label_offset = 0.05
        label_y = (vert_top_y + vert_bottom_y) / 2
        ax.text(vert_x + label_offset, label_y, output_label,
                ha='left', va='center', fontsize=12, rotation=90)

    return vert_x



def draw_dual_fuse(ax, x_left, y_center, left_term, right_term, left_input_left=None, left_input_right=None, left_output_left=None, left_output_right=None, left_input_connected='N', left_output_connected='N', right_input_left=None, right_input_right=None, right_output_left=None, right_output_right=None, right_input_connected='N', right_output_connected='N'):
    INNER_SPACING_MULT = 2.8
    inner_spacing = SYMBOL_WIDTH * INNER_SPACING_MULT
    x_right = x_left + inner_spacing
    def _draw_one_s(ax, x_pos, y_c, term, input_left, input_right, output_left, output_right, input_conn, output_conn, term_shift=0.0):
        fuse_top = y_c + SYMBOL_HEIGHT / 2
        fuse_bottom = y_c - SYMBOL_HEIGHT / 2
        top_circle_radius = SYMBOL_RADIUS * 0.8
        bottom_circle_radius = SYMBOL_RADIUS * 0.8
        ax.add_patch(Circle((x_pos, fuse_top), top_circle_radius,
                            edgecolor='black', facecolor='white', linewidth=1))
        ax.add_patch(Circle((x_pos, fuse_bottom), bottom_circle_radius,
                            edgecolor='black', facecolor='white', linewidth=1))
        start = (x_pos, fuse_top - top_circle_radius)
        end = (x_pos, fuse_bottom + bottom_circle_radius)
        ctrl1 = (x_pos + SYMBOL_RADIUS * 2.2, y_c + SYMBOL_HEIGHT * 0.15)
        ctrl2 = (x_pos - SYMBOL_RADIUS * 2.2, y_c - SYMBOL_HEIGHT * 0.15)
        t = np.linspace(0, 1, 100)
        xs = (1 - t)**3 * start[0] + 3 * (1 - t)**2 * t * ctrl1[0] + \
             3 * (1 - t) * t**2 * ctrl2[0] + t**3 * end[0]
        ys = (1 - t)**3 * start[1] + 3 * (1 - t)**2 * t * ctrl1[1] + \
             3 * (1 - t) * t**2 * ctrl2[1] + t**3 * end[1]
        ax.plot(xs, ys, color='black', linewidth=1, solid_capstyle='round')
        if pd.notna(term) and str(term).strip() != '':
            term_str = str(term)
            if term_str.endswith('.0'):
                term_str = term_str[:-2]
            text_x = x_pos + term_shift
            ax.text(text_x, y_c, term_str,
                    ha='center', va='center', fontsize=12)
        label_offset = 0.11
        if pd.notna(input_left) and str(input_left).strip() != "":
            ax.text(x_pos - label_offset + 0.11, fuse_top + 0.24, str(input_left),
                    fontsize=12, ha='right', va='bottom', rotation=90)
        if pd.notna(input_right) and str(input_right).strip() != "":
            ax.text(x_pos + 0.1, fuse_top + 0.27, str(input_right),
                    fontsize=12, ha='center', va='bottom', rotation=90)
        if pd.notna(output_left) and str(output_left).strip() != "":
            ax.text(x_pos - label_offset + 0.11, fuse_bottom - 0.30, str(output_left),
                    fontsize=12, ha='right', va='top', rotation=90)
        if pd.notna(output_right) and str(output_right).strip() != "":
            ax.text(x_pos + label_offset - 0.09, fuse_bottom - 0.28, str(output_right),
                    fontsize=12, ha='left', va='top', rotation=90)
        top_conn = (x_pos, fuse_top + top_circle_radius)
        bottom_conn = (x_pos, fuse_bottom - bottom_circle_radius)
        ic = 'Y' if str(input_conn).strip().upper() == 'Y' else 'N'
        oc = 'Y' if str(output_conn).strip().upper() == 'Y' else 'N'
        return top_conn, bottom_conn, ic, oc
    left_top, left_bottom, left_ic, left_oc = _draw_one_s(ax, x_left, y_center, left_term, left_input_left, left_input_right, left_output_left, left_output_right, left_input_connected, left_output_connected, term_shift=-0.1)
    right_top, right_bottom, right_ic, right_oc = _draw_one_s(ax, x_right, y_center, right_term, right_input_left, right_input_right, right_output_left, right_output_right, right_input_connected, right_output_connected, term_shift=-0.1)
    rail_extension = 0.15
    top_rail_y = max(left_top[1], right_top[1]) + rail_extension
    bottom_rail_y = min(left_bottom[1], right_bottom[1]) - rail_extension
    ax.plot([x_left, x_right], [top_rail_y, top_rail_y], linewidth=1, color='black')
    ax.plot([x_left, x_left], [left_top[1], top_rail_y], linewidth=1, color='black')
    ax.plot([x_right, x_right], [right_top[1], top_rail_y], linewidth=1, color='black')
    ax.plot([x_left, x_right], [bottom_rail_y, bottom_rail_y], linewidth=1, color='black')
    ax.plot([x_left, x_left], [bottom_rail_y, left_bottom[1]], linewidth=1, color='black')
    ax.plot([x_right, x_right], [bottom_rail_y, right_bottom[1]], linewidth=1, color='black')
    top_conn = (x_left, top_rail_y)
    bottom_conn = (x_left, bottom_rail_y)
    return top_conn, bottom_conn, left_ic, left_oc, right_ic, right_oc

def draw_resistor(ax, x, y_center, input_terminal='', output_terminal='', resistor_name='R', input_x_pos=None, output_x_pos=None):
    radius = SYMBOL_RADIUS * 1.5
    ax.add_patch(Circle((x, y_center), radius, edgecolor='black', facecolor='white', linewidth=1))
    ax.text(x, y_center, resistor_name, ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Upper vertical line
    upper_y_start = y_center + radius
    upper_y_end = y_center + radius * 7.5
    ax.plot([x, x], [upper_y_start, upper_y_end], color='black', linewidth=1)
    
    # Upper horizontal lines (dynamic to multiple input_x_pos if provided) with vertical drop at start
    upper_labels = [label.strip() for label in str(input_terminal).strip().split(',') if label.strip()]
    if input_x_pos is not None and isinstance(input_x_pos, (list, tuple)) and upper_labels:
        for i in range(len(upper_labels)):
            label = upper_labels[i]
            pos = input_x_pos[i] if i < len(input_x_pos) else x
            if i == 0:
                left_x = min(pos, x)
            else:
                left_x = min(input_x_pos[i-1], pos) if i > 0 and i < len(input_x_pos) else min(pos, x)
            right_x = max(pos, x) if i == len(upper_labels) - 1 else (input_x_pos[i + 1] if i + 1 < len(input_x_pos) else x)
            ax.plot([left_x, right_x], [upper_y_end, upper_y_end], color='black', linewidth=1)
            # Add small vertical line downward from the left end
            vertical_drop_length = 0.5
            ax.plot([left_x, left_x], [upper_y_end, upper_y_end - vertical_drop_length], color='black', linewidth=1)
            if label:
                ax.text((left_x + right_x) / 2, upper_y_end + 0.05, label, ha='center', va='bottom', fontsize=12)
    else:
        upper_horiz_length = 0.5 + len(str(input_terminal).strip()) * 0.12
        ax.plot([x - upper_horiz_length, x], [upper_y_end, upper_y_end], color='black', linewidth=1)
    
    # Lower vertical line
    # Lower vertical line
    lower_y_start = y_center - radius
    lower_y_end = y_center - radius * 6
    ax.plot([x, x], [lower_y_start, lower_y_end], color='black', linewidth=1)
    
    # Lower horizontal lines (dynamic to multiple output_x_pos if provided) with vertical rise at start
    lower_labels = [label.strip() for label in str(output_terminal).strip().split(',') if label.strip()]
    if output_x_pos is not None and isinstance(output_x_pos, (list, tuple)) and lower_labels:
        for i in range(len(lower_labels)):
            label = lower_labels[i]
            pos = output_x_pos[i] if i < len(output_x_pos) else x
            if i == 0:
                left_x = min(pos, x)
            else:
                left_x = min(output_x_pos[i-1], pos) if i > 0 and i < len(output_x_pos) else min(pos, x)
            right_x = max(pos, x) if i == len(lower_labels) - 1 else (output_x_pos[i + 1] if i + 1 < len(output_x_pos) else x)
            ax.plot([left_x, right_x], [lower_y_end, lower_y_end], color='black', linewidth=1)
            # Add small vertical line upward from the left end
            vertical_drop_length = 0.5
            ax.plot([left_x, left_x], [lower_y_end, lower_y_end + vertical_drop_length], color='black', linewidth=1)
            if label:
                ax.text((left_x + right_x) / 2, lower_y_end - 0.05, label, ha='center', va='top', fontsize=12)
    else:
        lower_horiz_length = 0.5 + len(str(output_terminal).strip()) * 0.12
        ax.plot([x - lower_horiz_length, x], [lower_y_end, lower_y_end], color='black', linewidth=1)
    
    return None, None



def draw_input_connection(ax, x, symbol_top_y, connected_flag, y_top_bus_group):
    overlap = SYMBOL_RADIUS * 0.15
    start_y = symbol_top_y - overlap
    ax.plot([x, x], [start_y, y_top_bus_group], color='black', linewidth=1)
    return True

def draw_output_connection(ax, x, symbol_bottom_y, connected_flag, y_bottom_bus_group):
    overlap = SYMBOL_RADIUS * 0.15
    start_y = symbol_bottom_y + overlap
    ax.plot([x, x], [start_y, y_bottom_bus_group], color='black', linewidth=1)
    return True

def draw_bus_lines(ax, x_positions, connected_flags, bus_y, gap=0.12, extra=0.12):
    if not x_positions:
        return
    segments = []
    start_idx = None
    for i, connected in enumerate(connected_flags):
        if connected:
            if start_idx is None:
                start_idx = i
        else:
            if start_idx is not None:
                segments.append((start_idx, i-1))
                start_idx = None
    if start_idx is not None:
        segments.append((start_idx, len(connected_flags)-1))
    for start_idx, end_idx in segments:
        x_start = x_positions[start_idx]
        x_end = x_positions[end_idx]
        if start_idx == end_idx:
            pass
        else:
            total_len = x_end - x_start
            shrink = min(gap, total_len / 4.0)
            plot_start = x_start + shrink - extra
            plot_end = x_end - shrink + extra
            if plot_end <= plot_start:
                mid = (x_start + x_end) / 2.0
                small = max(0.06, gap)
                ax.plot([mid - small, mid + small], [bus_y, bus_y], color='black', linewidth=1)
            else:
                ax.plot([plot_start, plot_end], [bus_y, bus_y], color='black', linewidth=1)

# === Helper function for parsing terminal_no field ===
def parse_terminal_no_field(val):
    if pd.isna(val):
        return None, None
    s = str(val).strip()
    if ',' in s:
        parts = s.split(',')
        if len(parts) >= 2:
            try:
                a = parts[0].strip()
                b = parts[1].strip()
                return a, b
            except ValueError:
                pass
    if '-' in s:
        parts = s.split('-')
        if len(parts) >= 2:
            try:
                a = parts[0].strip()
                b = parts[1].strip()
                return a, b
            except ValueError:
                pass
    return s, s

# === Main Draw Function ===
def draw_symbols(df, ax, ordered_circuit_ids, junction_name, start_x=1, pin_spacing=0.8, circuits_per_page=12, page_number=1, max_terminal_symbols_per_row=36, max_rows_visible=3, page_width=None):
    """
    Draw symbols for the provided ordered_circuit_ids on ax.
    max_terminal_symbols_per_row: maximum number of terminal symbols per row (default 36).
    max_rows_visible: maximum number of visible symbol rows per page (default 3).
    If drawing would start a 4th row, that row is reserved as blank and remaining circuits for the page are not drawn.
    """
    current_x = start_x
    current_terminal_count = 0  # Count terminal symbols instead of circuits
    y_offset = 0
    all_x_positions = []
    all_input_connected_flags = []
    all_output_connected_flags = []
    min_y = float('inf')
    max_y = float('-inf')
    overall_max_x = start_x
    current_row_max_x = start_x

    # Track which logical row index we are on (0-based)
    row_index = 0
    stop_drawing = False

    # Use page-specific circuit ids supplied (ordered already per-junction)
    page_circuit_ids = ordered_circuit_ids

    # Group circuits by circuit letter (preserve order of first appearance)
    letter_groups = OrderedDict()
    for cid in page_circuit_ids:
        r = df_circuit[df_circuit['circuit_id'] == cid]
        letter = ""
        if not r.empty and 'circuit_name' in r.columns:
            cn = str(r['circuit_name'].iloc[0]).strip()
            m = re.match(r'^([A-Z])', cn)
            if m:
                letter = m.group(1)
            else:
                # fallback to whole name if no leading uppercase letter
                letter = cn if cn else ""
        else:
            letter = ""
        if letter not in letter_groups:
            letter_groups[letter] = []
        letter_groups[letter].append(cid)

    # Iterate letter groups so each letter starts at a new row
    for letter, circuit_list in letter_groups.items():
        if stop_drawing:
            break

        # If we're mid-row when starting a new letter, force a new row (this leaves blanks at end of prior letter)
        if current_terminal_count != 0:
            # about to start a new row
            # increment row index and check visible limit
            row_index += 1
            overall_max_x = max(overall_max_x, current_row_max_x)
            current_row_max_x = start_x
            current_x = start_x
            current_terminal_count = 0
            # Reserve space for new row (visible or blank)
            y_offset -= vertical_gap

            if row_index >= max_rows_visible:
                # we've reached the 4th (or beyond) row: reserve the blank row and stop drawing further circuits on this page
                stop_drawing = True
                break

        # set group label to draw at row start (user wants letter at left for each row)
        group_label = letter if letter else " "

        for circuit_id in circuit_list:
            if stop_drawing:
                break

            circuit_rows = df_circuit[df_circuit['circuit_id'] == circuit_id]
            circuit_pos = circuit_rows['position'].iloc[0] if not circuit_rows.empty and 'position' in circuit_rows.columns else None
            group = df_symbols[df_symbols['circuit_id'] == circuit_id].sort_index().reset_index(drop=True)

            # Determine capsule center for this row
            capsule_y_center = CAPSULE_Y_CENTER_BASE + y_offset
            y_top_bus_group = capsule_y_center + y_top_bus_offset
            y_bottom_bus_group = capsule_y_center + y_bottom_bus_offset

            if group.empty:
                current_x += pin_spacing + CIRCUIT_GAP
                current_row_max_x = max(current_row_max_x, current_x)
                current_terminal_count += 1  # Count as one for empty groups
                min_y = min(min_y, capsule_y_center - 2.0)
                max_y = max(max_y, capsule_y_center + 2.0)
                continue

            input_connected_flags = []
            output_connected_flags = []
            x_positions = []
            symbol_bottoms = []
            terminal_names_for_positions = []

            i = 0
            while i < len(group) and not stop_drawing:
                row = group.iloc[i]
                symbol = str(row.get('symbol', '')).strip().lower()
                symbols_to_add = 2 if symbol == 'dual_fuse' else 1

                # Check if adding the next symbol(s) would exceed max_terminal_symbols_per_row
                if current_terminal_count + symbols_to_add > max_terminal_symbols_per_row:
                    # Start new row
                    row_index += 1
                    overall_max_x = max(overall_max_x, current_row_max_x)
                    current_row_max_x = start_x
                    current_x = start_x
                    current_terminal_count = 0
                    y_offset -= vertical_gap
                    capsule_y_center = CAPSULE_Y_CENTER_BASE + y_offset
                    y_top_bus_group = capsule_y_center + y_top_bus_offset
                    y_bottom_bus_group = capsule_y_center + y_bottom_bus_offset

                    if row_index >= max_rows_visible:
                        stop_drawing = True
                        break

                if current_terminal_count == 0 and not stop_drawing:
                    # draw the group letter (A, B, ...) at the beginning of each visible row
                    draw_circuit_name(ax, start_x - 1.2, capsule_y_center, group_label or f"Circuit {circuit_id}")

                if stop_drawing:
                    break

                if symbol == 'capsule':
                    top_conn, bottom_conn, input_conn, output_conn = draw_capsule(
                        ax, current_x, capsule_y_center,
                        row.get('terminal_name'),
                        row.get('input_left'),
                        row.get('input_right'),
                        row.get('output_left'),
                        row.get('output_right'),
                        row.get('input_connected', 'N'),
                        row.get('output_connected', 'N')
                    )
                    x_positions.append(current_x)
                    tname = str(row.get('terminal_name')).strip()
                    if tname.endswith('.0'):
                        tname = tname[:-2]
                    terminal_names_for_positions.append(tname)
                    input_connected_flags.append(str(input_conn).strip().upper() == "Y")
                    output_connected_flags.append(str(output_conn).strip().upper() == "Y")
                    symbol_bottoms.append(capsule_y_center - SYMBOL_HEIGHT / 2 - SYMBOL_RADIUS)
                    current_x += pin_spacing
                    current_row_max_x = max(current_row_max_x, current_x)
                    current_terminal_count += 1
                    i += 1
                elif symbol == 'single_fuse':
                    top_conn, bottom_conn, input_conn, output_conn = draw_s_fuse(
                        ax, current_x, capsule_y_center, row.get('terminal_name'),
                        row.get('input_left'), row.get('input_right'), row.get('output_left'), row.get('output_right'),
                        row.get('input_connected', 'N'), row.get('output_connected', 'N')
                    )
                    x_positions.append(current_x)
                    tname = str(row.get('terminal_name')).strip()
                    if tname.endswith('.0'):
                        tname = tname[:-2]
                    terminal_names_for_positions.append(tname)
                    input_connected_flags.append(str(input_conn).strip().upper() == "Y")
                    output_connected_flags.append(str(output_conn).strip().upper() == "Y")
                    symbol_bottoms.append(capsule_y_center - SYMBOL_HEIGHT / 2 - SYMBOL_RADIUS)
                    current_x += pin_spacing
                    current_row_max_x = max(current_row_max_x, current_x)
                    current_terminal_count += 1
                    i += 1
                if symbol == 'choke':
                    i += 1
                    continue  # Skip vertical symbol drawing; we'll handle horizontal later
                elif symbol == 'dual_fuse':
                    ##
                    if i + 1 < len(group):
                        next_row = group.iloc[i+1]
                        LEFT_EXTRA = pin_spacing * 1.0
                        AFTER_SPACING = pin_spacing * 1.5
                        current_x += LEFT_EXTRA
                        dual_start_x = current_x - SYMBOL_WIDTH * 1.25
                        top_conn, bottom_conn, left_ic, left_oc, right_ic, right_oc = draw_dual_fuse(
                            ax, dual_start_x, capsule_y_center,
                            row.get('terminal_name'),
                            next_row.get('terminal_name'),
                            row.get('input_left'), row.get('input_right'), row.get('output_left'), row.get('output_right'),
                            row.get('input_connected', 'N'), row.get('output_connected', 'N'),
                            next_row.get('input_left'), next_row.get('input_right'), next_row.get('output_left'), next_row.get('output_right'),
                            next_row.get('input_connected', 'N'), next_row.get('output_connected', 'N')
                        )
                        tname_left = str(row.get('terminal_name')).strip()
                        tname_right = str(next_row.get('terminal_name')).strip()
                        if tname_left.endswith('.0'): tname_left = tname_left[:-2]
                        if tname_right.endswith('.0'): tname_right = tname_right[:-2]
                        x_positions.append(dual_start_x)
                        x_positions.append(dual_start_x)
                        terminal_names_for_positions.append(tname_left)
                        terminal_names_for_positions.append(tname_right)
                        input_connected_flags.append(str(left_ic).strip().upper() == "Y")
                        input_connected_flags.append(str(right_ic).strip().upper() == "Y")
                        output_connected_flags.append(str(left_oc).strip().upper() == "Y")
                        output_connected_flags.append(str(right_oc).strip().upper() == "Y")
                        symbol_bottoms.append(capsule_y_center - SYMBOL_HEIGHT / 2 - SYMBOL_RADIUS)
                        symbol_bottoms.append(capsule_y_center - SYMBOL_HEIGHT / 2 - SYMBOL_RADIUS)
                        current_x += AFTER_SPACING
                        current_row_max_x = max(current_row_max_x, current_x)
                        current_terminal_count += 2
                        i += 2
                    else:
                        top_conn, bottom_conn, input_conn, output_conn = draw_s_fuse(
                            ax, current_x, capsule_y_center, row.get('terminal_name'),
                            row.get('input_left'), row.get('input_right'), row.get('output_left'), row.get('output_right'),
                            row.get('input_connected', 'N'), row.get('output_connected', 'N')
                        )
                        x_positions.append(current_x)
                        tname = str(row.get('terminal_name')).strip()
                        if tname.endswith('.0'):
                            tname = tname[:-2]
                        terminal_names_for_positions.append(tname)
                        input_connected_flags.append(str(input_conn).strip().upper() == "Y")
                        output_connected_flags.append(str(output_conn).strip().upper() == "Y")
                        symbol_bottoms.append(capsule_y_center - SYMBOL_HEIGHT / 2 - SYMBOL_RADIUS)
                        current_x += pin_spacing
                        current_row_max_x = max(current_row_max_x, current_x)
                        current_terminal_count += 1
                        i += 1
            # Add middle space
            current_x += pin_spacing  # extra space after symbols
            current_row_max_x = max(current_row_max_x, current_x)
            # Add resistor if applicable
            resistor_row = df_resistor[df_resistor['circuit_id'] == circuit_id]
            special_resistor = False
            if not resistor_row.empty and str(resistor_row['resistor'].iloc[0]).strip().lower() == 'yes':
                special_resistor = True
                resistor_label = str(resistor_row['resistor_name'].iloc[0]).strip() if 'resistor_name' in resistor_row.columns and pd.notna(resistor_row['resistor_name'].iloc[0]) else 'R'
                input_terms = [term.strip() for term in str(resistor_row['input_terminal'].iloc[0]).strip().replace('.0', '').split(',') if term.strip()]
                output_terms = [term.strip() for term in str(resistor_row['output_terminal'].iloc[0]).strip().replace('.0', '').split(',') if term.strip()]
                input_x_pos = [x_positions[terminal_names_for_positions.index(term)] for term in input_terms if term in terminal_names_for_positions] if input_terms else None
                output_x_pos = [x_positions[terminal_names_for_positions.index(term)] for term in output_terms if term in terminal_names_for_positions] if output_terms else None
                symbols_to_add = 1
                if current_terminal_count + symbols_to_add > max_terminal_symbols_per_row:
                    row_index += 1
                    overall_max_x = max(overall_max_x, current_row_max_x)
                    current_row_max_x = start_x
                    current_x = start_x
                    current_terminal_count = 0
                    y_offset -= vertical_gap
                    capsule_y_center = CAPSULE_Y_CENTER_BASE + y_offset
                    y_top_bus_group = capsule_y_center + y_top_bus_offset
                    y_bottom_bus_group = capsule_y_center + y_bottom_bus_offset
                    if row_index >= max_rows_visible:
                        stop_drawing = True
                        break
                current_x -= 0.5
                draw_resistor(ax, current_x, capsule_y_center, input_terminal=','.join(input_terms) if input_terms else '', output_terminal=','.join(output_terms) if output_terms else '', resistor_name=resistor_label, input_x_pos=input_x_pos, output_x_pos=output_x_pos)
                current_x += pin_spacing
                current_row_max_x = max(current_row_max_x, current_x)
                current_terminal_count += 1


            # If we reserved a blank 4th row mid-way, skip the rest
            if stop_drawing:
                break

            # Draw horizontal choke on bottom bus if specified in choketable
            choke_row = df_choke[df_choke['circuit_id'] == circuit_id]
            special_choke = False
            vert_x = None
            if not choke_row.empty and str(choke_row['choke'].iloc[0]).strip().lower() == 'yes':
                input_term = str(choke_row['input_terminal'].iloc[0]).strip().replace('.0', '')
                output_term = str(choke_row['output_terminal'].iloc[0]).strip().replace('.0', '')
                if input_term in terminal_names_for_positions:
                    start_idx = terminal_names_for_positions.index(input_term)
                    x_left = x_positions[start_idx]
                    choke_label = str(choke_row['terminal_name'].iloc[0]).strip() if 'terminal_name' in choke_row.columns and pd.notna(choke_row['terminal_name'].iloc[0]) else 'CHOKE'
                    if output_term in terminal_names_for_positions:
                        end_idx = terminal_names_for_positions.index(output_term)
                        x_right = x_positions[end_idx]
                        box_width = max(1.2, x_right - x_left - 0.2)
                        draw_horizontal_choke(ax, (x_left + x_right) / 2, y_bottom_bus_group, label=choke_label, box_width=box_width)
                    else:
                        special_choke = True
                        box_width = 1.2
                        x_center = x_left + 0.8
                        vert_x = draw_horizontal_choke(ax, x_center, y_bottom_bus_group, label=choke_label, box_width=box_width, special_end=True, output_label=output_term)

            hook_input_flags = []
            hook_output_flags = []
            for j, x in enumerate(x_positions):
                if j > 0 and x == x_positions[j-1]:
                    continue
                symbol_top_y = capsule_y_center + SYMBOL_HEIGHT/2 + SYMBOL_RADIUS
                symbol_bottom_y = capsule_y_center - SYMBOL_HEIGHT/2 - SYMBOL_RADIUS
                hooked_in = draw_input_connection(ax, x, symbol_top_y, 'Y' if input_connected_flags[j] else 'N', y_top_bus_group)
                hooked_out = draw_output_connection(ax, x, symbol_bottom_y, 'Y' if output_connected_flags[j] else 'N', y_bottom_bus_group)
                hook_input_flags.append(hooked_in)
                hook_output_flags.append(hooked_out)

            top_ranges = []
            bottom_ranges = []
            circuit_headers_temp = df_header[df_header['circuit_id'] == circuit_id]
            for _, hrow_temp in circuit_headers_temp.iterrows():
                header_type_temp = str(hrow_temp.get('header_type', '')).strip().upper()
                terminal_start_temp = hrow_temp.get('terminal_start')
                terminal_end_temp = hrow_temp.get('terminal_end', terminal_start_temp)
                start_name_temp = str(terminal_start_temp).strip().replace('.0', '') if pd.notna(terminal_start_temp) else None
                end_name_temp = str(terminal_end_temp).strip().replace('.0', '') if pd.notna(terminal_end_temp) else None
                if pd.isna(start_name_temp) or pd.isna(end_name_temp) or start_name_temp not in terminal_names_for_positions or end_name_temp not in terminal_names_for_positions:
                    continue
                start_idx_temp = terminal_names_for_positions.index(start_name_temp)
                end_idx_temp = terminal_names_for_positions.index(end_name_temp)
                if start_idx_temp > end_idx_temp:
                    start_idx_temp, end_idx_temp = end_idx_temp, start_idx_temp
                if header_type_temp == 'WIREFROM':
                    top_ranges.append((start_idx_temp, end_idx_temp))
                elif header_type_temp == 'WIRETO':
                    bottom_ranges.append((start_idx_temp, end_idx_temp))

            merge_adjacent = True
            if top_ranges and bottom_ranges:
                merge_adjacent = False

            top_segments = merge_ranges(top_ranges, merge_adjacent=merge_adjacent)
            bottom_segments = merge_ranges(bottom_ranges, merge_adjacent=merge_adjacent)

            if not top_segments and any(input_connected_flags):
                top_segments = [(0, len(x_positions)-1)]
            if not bottom_segments and any(output_connected_flags):
                bottom_segments = [(0, len(x_positions)-1)]

            for min_idx, max_idx in top_segments:
                sub_x = x_positions[min_idx : max_idx + 1]
                sub_flags = input_connected_flags[min_idx : max_idx + 1]
                draw_bus_lines(ax, sub_x, sub_flags, y_top_bus_group, gap=0.12)
                connected_local = [i for i, f in enumerate(sub_flags) if f]
                if connected_local:
                    first_local = connected_local[0]
                    x_first = sub_x[first_local]
                    ax.plot([x_first - 0.3, x_first], [y_top_bus_group, y_top_bus_group], color='black', linewidth=1)
                    ax.plot([x_first - 0.3, x_first - 0.3], [y_top_bus_group, y_top_bus_group + 0.2], color='black', linewidth=1)

            for min_idx, max_idx in bottom_segments:
                sub_x = x_positions[min_idx : max_idx + 1]
                sub_flags = output_connected_flags[min_idx : max_idx + 1]
                draw_bus_lines(ax, sub_x, sub_flags, y_bottom_bus_group, gap=0.12)
                connected_local = [i for i, f in enumerate(sub_flags) if f]
                if connected_local:
                    last_local = connected_local[-1]
                    x_last = sub_x[last_local]
                    if (special_choke and max_idx == start_idx and min_idx <= start_idx) or special_resistor:
                        # skip standard hook for special choke or resistor
                        pass
                    else:
                        ax.plot([x_last, x_last + 0.3], [y_bottom_bus_group, y_bottom_bus_group], color='black', linewidth=1)
                        ax.plot([x_last + 0.3, x_last + 0.3], [y_bottom_bus_group, y_bottom_bus_group - 0.2], color='black', linewidth=1)

            circuit_groups = df_group[df_group['circuit_id'] == circuit_id] if 'circuit_id' in df_group.columns else pd.DataFrame()
            name_to_x = {}
            name_to_output_connected = {}
            name_to_input_connected = {}
            for idx, (xval, tname) in enumerate(zip(x_positions, terminal_names_for_positions)):
                if tname in name_to_x:
                    continue
                name_to_x[tname] = xval
                name_to_output_connected[tname] = output_connected_flags[idx] if idx < len(output_connected_flags) else False
                name_to_input_connected[tname] = input_connected_flags[idx] if idx < len(input_connected_flags) else False

            x_min = min(x_positions) if x_positions else None
            x_max = max(x_positions) if x_positions else None

            if not circuit_groups.empty:
                min_bottom = min(symbol_bottoms) if symbol_bottoms else y_bottom_bus_group
                x_start_pos = min(x_positions) if x_positions else current_x - pin_spacing
                x_end_pos = max(x_positions) if x_positions else current_x

                for _, grow in circuit_groups.iterrows():
                    tn_field = grow.get('terminal_no')
                    start_name, end_name = parse_terminal_no_field(tn_field)
                    x_start_term = name_to_x.get(start_name, x_min)
                    x_end_term = name_to_x.get(end_name, x_max)
                    if x_start_term is None or x_end_term is None:
                        continue
                    label_text = grow.get('text', '')
                    io_field = str(grow.get('input_output', '')).strip().lower()
                    if io_field == 'input':
                        y_relay = y_top_bus_group + 0.55
                        draw_relay_input(ax, x_start_term, x_end_term, y=y_relay, scale=1.0, text=str(label_text))
                    elif io_field == 'output':
                        y_relay = y_bottom_bus_group - 0.55
                        draw_relay_output(ax, x_start_term, x_end_term, y=y_relay, scale=1.0, text=str(label_text))
                    else:
                        center_x = (x_start_term + x_end_term) / 2.0
                        ax.text(center_x, y_top_bus_group + 0.2, str(label_text), ha='center', va='bottom', fontsize=8, fontweight='bold')

            circuit_headers = df_header[df_header['circuit_id'] == circuit_id]
            relay_top = {}
            relay_bottom = {}
            for _, hrow in circuit_headers.iterrows():
                header_type = str(hrow.get('header_type', '')).strip().upper()
                terminal_start = hrow.get('terminal_start')
                terminal_end = hrow.get('terminal_end', terminal_start)
                input_output = str(hrow.get('input_output', '')).strip().lower()
                text = hrow.get('text', '')
                if pd.isna(text) or str(text).strip() == '':
                    text = ''
                else:
                    text = str(text).strip()
                start_name = str(terminal_start).strip().replace('.0', '') if pd.notna(terminal_start) else None
                end_name = str(terminal_end).strip().replace('.0', '') if pd.notna(terminal_end) else None
                if pd.isna(start_name) or pd.isna(end_name) or start_name not in terminal_names_for_positions or end_name not in terminal_names_for_positions:
                    continue
                start_idx_temp = terminal_names_for_positions.index(start_name)
                end_idx_temp = terminal_names_for_positions.index(end_name)
                if start_idx_temp > end_idx_temp:
                    start_idx_temp, end_idx_temp = end_idx_temp, start_idx_temp
                x_left = x_positions[start_idx_temp]
                x_right = x_positions[end_idx_temp]

                if header_type == 'RELAY':
                    terminal_start_str = str(terminal_start).strip().replace('.0', '') if pd.notna(terminal_start) else None
                    terminal_end_str = str(terminal_end).strip().replace('.0', '') if pd.notna(terminal_end) else None
                    if terminal_start_str is None or terminal_end_str is None:
                        continue
                    key = (circuit_id, terminal_start_str, terminal_end_str)
                    text = str(hrow.get('text', '')).strip()
                    input_output = str(hrow.get('input_output', '')).strip().lower()
                    if input_output == 'input':
                        if key not in relay_top:
                            relay_top[key] = []
                        relay_top[key].append(text)
                    elif input_output == 'output':
                        if key not in relay_bottom:
                            relay_bottom[key] = []
                        relay_bottom[key].append(text)
                    continue
                elif header_type in ['WIREFROM', 'WIRETO']:
                    min_symbol_bottom_local = min(symbol_bottoms[start_idx_temp:end_idx_temp+1]) if symbol_bottoms else None
                    if header_type == 'WIREFROM':
                        sub_flags = input_connected_flags[start_idx_temp:end_idx_temp+1]
                        connected_local = [i for i, f in enumerate(sub_flags) if f]
                        first_hook_x_specific = x_positions[start_idx_temp + connected_local[0]] if connected_local else x_left
                        draw_header(ax, circuit_id, header_type, x_left, x_right, text,
                                    min_symbol_bottom=min_symbol_bottom_local,
                                    first_hook_x=first_hook_x_specific,
                                    last_hook_x=None,
                                    y_top_bus_group=y_top_bus_group,
                                    y_bottom_bus_group=y_bottom_bus_group)
                    elif header_type == 'WIRETO':
                        sub_flags = output_connected_flags[start_idx_temp:end_idx_temp+1]
                        connected_local = [i for i, f in enumerate(sub_flags) if f]
                        last_hook_x_specific = x_positions[start_idx_temp + connected_local[-1]] if connected_local else None
                        special_ha_local = False
                        if special_choke and end_idx_temp == start_idx:
                            last_hook_x_specific = vert_x
                            special_ha_local = True
                        draw_header(ax, circuit_id, header_type, x_left, x_right, text,
                                    min_symbol_bottom=min_symbol_bottom_local,
                                    first_hook_x=None,
                                    last_hook_x=last_hook_x_specific,
                                    y_top_bus_group=y_top_bus_group,
                                    y_bottom_bus_group=y_bottom_bus_group,
                                    special_ha=special_ha_local)
            for key, texts in relay_top.items():
                if not texts:
                    continue
                cid, start_name, end_name = key
                if start_name not in terminal_names_for_positions or end_name not in terminal_names_for_positions:
                    continue
                start_idx_temp = terminal_names_for_positions.index(start_name)
                end_idx_temp = terminal_names_for_positions.index(end_name)
                if start_idx_temp > end_idx_temp:
                    start_idx_temp, end_idx_temp = end_idx_temp, start_idx_temp
                x_left = x_positions[start_idx_temp]
                x_right = x_positions[end_idx_temp]
                symbol_top_y = capsule_y_center + SYMBOL_HEIGHT/2 + SYMBOL_RADIUS
                input_conn_flag = any(name_to_input_connected.get(term, False) for term in terminal_names_for_positions[start_idx_temp:end_idx_temp+1])
                vertical_line_start = y_top_bus_group if input_conn_flag else symbol_top_y + stub_length
                draw_group_top_symbol(ax, x_left, x_right, vertical_line_start, texts=texts, scale=1.0, input_connected='Y' if input_conn_flag else 'N')

            for key, texts in relay_bottom.items():
                if not texts:
                    continue
                cid, start_name, end_name = key
                if start_name not in terminal_names_for_positions or end_name not in terminal_names_for_positions:
                    continue
                start_idx_temp = terminal_names_for_positions.index(start_name)
                end_idx_temp = terminal_names_for_positions.index(end_name)
                if start_idx_temp > end_idx_temp:
                    start_idx_temp, end_idx_temp = end_idx_temp, start_idx_temp
                x_left = x_positions[start_idx_temp]
                x_right = x_positions[end_idx_temp]
                symbol_bottom_y = capsule_y_center - SYMBOL_HEIGHT/2 - SYMBOL_RADIUS
                output_conn_flag = any(name_to_output_connected.get(term, False) for term in terminal_names_for_positions[start_idx_temp:end_idx_temp+1])
                vertical_line_end = y_bottom_bus_group if output_conn_flag else symbol_bottom_y - stub_length
                choke_output_terminal = None
                choke_info = df_choke[df_choke['circuit_id'] == cid]
                if not choke_info.empty:
                    output_terminal = str(choke_info['output_terminal'].iloc[0]).strip()
                    if output_terminal.endswith('.0'):
                        output_terminal = output_terminal[:-2]
                    if output_terminal in [start_name, end_name]:
                        choke_output_terminal = output_terminal
                draw_group_bottom_symbol(ax, x_left, x_right, vertical_line_end, text=texts[0], 
                                        output_connected='Y' if output_conn_flag else 'N', 
                                        choke_output_terminal=choke_output_terminal)

            all_x_positions.extend(x_positions)
            all_input_connected_flags.extend(input_connected_flags)
            all_output_connected_flags.extend(output_connected_flags)
            current_x += CIRCUIT_GAP
            current_row_max_x = max(current_row_max_x, current_x)
            min_y = min(min_y, y_bottom_bus_group - 1.8)
            max_y = max(max_y, y_top_bus_group + 1.8)

    overall_max_x = max(overall_max_x, current_row_max_x)

    # Determine content min/max from recorded positions (fallbacks provided)
    if all_x_positions:
        content_min_x = min(all_x_positions)
        content_max_x = max(all_x_positions)
    else:
        content_min_x = start_x
        content_max_x = start_x + (page_width if page_width else fixed_fig_width)

    content_width = max(0.1, content_max_x - content_min_x)

    # Choose desired drawing width (units in the same data coordinates as x positions)
    # add a safety margin of 2.0 units so labels aren't clipped
    desired_width = global_max_width

    # Left and right limits (keep left fixed relative to start_x so rows align)
    left = start_x - 1.5
    right = left + desired_width

    # Calculate page center for junction box using fixed horizontal span
    page_center_x = (left + right) / 2.0

    ax.set_xlim(left, right)
    ax.set_ylim(fixed_ylim_min, fixed_ylim_max)

    # Draw manual horizontal line with fixed absolute coordinates
    manual_y = -12.65
    ax.plot([left - 2, right + 2], [manual_y, manual_y], 'k-', linewidth=2.0, zorder=10)




    # draw junction box at top using the computed page_center_x
    junction_box_y = CAPSULE_Y_CENTER_BASE + y_top_bus_offset +1.8 + 3.0 -1.0
    draw_junction_box(ax, page_center_x, junction_box_y, junction_name)

    return all_x_positions, all_input_connected_flags, all_output_connected_flags

# === Function to draw footer ===
def draw_footer(ax, left, right, fixed_ylim_min, total_pages, page_num, df_title_row, junction_name):
    """
    Draw a compact footer (title-block style) on `ax` occupying the right 50% of [left, right].
    All line widths are set to 1.3 and all text font sizes are set to 20 (as requested).
    """
    if df_title_row is None:
        return

    width = 20.0
    extra_width = 6.0
    height = 3.5  # adjusted footer height
    total_block_width = width + extra_width  # 26

    # Footer occupies right 50% at bottom
    footer_width = (right - left) / 2.0
    footer_x_start = left + footer_width
    footer_y_start = fixed_ylim_min
    base_height = 3.0  # original design height
    scale = height / base_height  # 1.5 / 3.0 = 0.5
    x_scale = footer_width / total_block_width
    LINEWIDTH = 1.3
    FONTSIZE = 12.5

    # helper to scale y-coordinates
    s = lambda y: y * scale

    outer_width = total_block_width * x_scale
    outer_height = height

    # Full outer rectangle
    ax.add_patch(Rectangle((footer_x_start, footer_y_start), outer_width, outer_height,
                           fill=False, linewidth=LINEWIDTH))

    # Vertical lines (major)
    v_lines = [8, 12, 17.5, width + extra_width / 2]
    for vx in v_lines:
        x = footer_x_start + vx * x_scale
        ax.plot([x, x], [footer_y_start, footer_y_start + outer_height], 'k-', linewidth=LINEWIDTH)

    # Special vertical at 14.8 that starts from y=1.5 (scaled)
    x14_8 = footer_x_start + 14.8 * x_scale
    ax.plot([x14_8, x14_8], [footer_y_start + s(1.5), footer_y_start + outer_height], 'k-', linewidth=LINEWIDTH)

    # Horizontal lines (scaled y positions)
    horizontal_lines = [
        (0, 8, s(1.5)),
        (0, 8, s(1.0)),
        (12, 17.5, s(2.0)),
        (23, 26, s(0.7)),
        (12, 26, s(1.5)),
        (12, 26, s(2.5))
    ]
    for x0, x1, y in horizontal_lines:
        ax.plot([footer_x_start + x0 * x_scale, footer_x_start + x1 * x_scale],
                [footer_y_start + y, footer_y_start + y],
                'k-', linewidth=LINEWIDTH)

    # Small vertical inner lines (x=4 from y=0 to 1.5; x=25.2 from y=-0.02 to 1.5)
    ax.plot([footer_x_start + 4 * x_scale, footer_x_start + 4 * x_scale],
            [footer_y_start + 0.0, footer_y_start + s(1.5)], 'k-', linewidth=LINEWIDTH)
    ax.plot([footer_x_start + 25.2 * x_scale, footer_x_start + 25.2 * x_scale],
            [footer_y_start + s(-0.02), footer_y_start + s(1.5)], 'k-', linewidth=LINEWIDTH)

    # Company text (slight y-shift to avoid clipping)
    company_y = footer_y_start + s(2.6) - 0.02
    ax.text(footer_x_start + 0.1 * x_scale, company_y,
            "M/S. YOLAX INFRAENERGY\nPRIVATE. LTD.,\nINDORE.",
            va='top', ha='left', fontsize=FONTSIZE, weight='bold', linespacing=1.2)

    # Labels
    ax.text(footer_x_start + 1.4 * x_scale, footer_y_start + s(0.5), "DRAWN BY",
            va='center', ha='left', fontsize=FONTSIZE, weight='bold')
    ax.text(footer_x_start + 5.4 * x_scale, footer_y_start + s(0.5), "CHECKED BY",
            va='center', ha='left', fontsize=FONTSIZE, weight='bold')

    # Names above labels (guard for missing keys)
    drawn_by = df_title_row.get('drawn_by') if hasattr(df_title_row, 'get') else df_title_row.get('drawn_by', None)
    checked_by = df_title_row.get('checked_by') if hasattr(df_title_row, 'get') else df_title_row.get('checked_by', None)

    if pd.notna(drawn_by):
        ax.text(footer_x_start + 1.4 * x_scale, footer_y_start + s(1.2),
                str(drawn_by), va='bottom', ha='left', fontsize=FONTSIZE, weight='bold')
    if pd.notna(checked_by):
        ax.text(footer_x_start + 5.4 * x_scale, footer_y_start + s(1.2),
                str(checked_by), va='bottom', ha='left', fontsize=FONTSIZE, weight='bold')

    # Designations
    ax.text(footer_x_start + 15 * x_scale, footer_y_start + s(2.75),
            str(df_title_row.get('designation1', '')), va='top', ha='left', fontsize=FONTSIZE, weight='bold')
    ax.text(footer_x_start + 15 * x_scale, footer_y_start + s(2.25),
            str(df_title_row.get('designation2', '')), va='top', ha='left', fontsize=FONTSIZE, weight='bold')
    ax.text(footer_x_start + 15 * x_scale, footer_y_start + s(1.82),
            str(df_title_row.get('designation3', '')), va='top', ha='left', fontsize=FONTSIZE, weight='bold')

    # Station info - Use junction_name instead of diagram_name
    ax.text(footer_x_start + 19 * x_scale, footer_y_start + s(2.75),
            str(df_title_row.get('station_name', '')), va='top', ha='left', fontsize=FONTSIZE, weight='bold')
    ax.text(footer_x_start + 18 * x_scale, footer_y_start + s(2.0),
            junction_name, va='top', ha='left', fontsize=FONTSIZE, weight='bold')  # Changed to use junction_name
    ax.text(footer_x_start + 18.5 * x_scale, footer_y_start + s(0.9),
            f"DRG. NO. {df_title_row.get('station_code', '')}", va='top', ha='left', fontsize=FONTSIZE, weight='bold')

    # Division & Zone
    ax.text(footer_x_start + 24 * x_scale, footer_y_start + s(2.8),
            str(df_title_row.get('zone', '')), va='top', ha='left', fontsize=FONTSIZE, weight='bold')
    ax.text(footer_x_start + 23.3 * x_scale, footer_y_start + s(2.0),
            f"{df_title_row.get('division', '')} DIVISION", va='top', ha='left', fontsize=FONTSIZE, weight='bold')

    # Total sheet (override with actual total_pages)
    ax.text(footer_x_start + 25.5 * x_scale, footer_y_start + s(0.4),
            str(total_pages), va='bottom', ha='left', fontsize=FONTSIZE, weight='bold')

    # Right-side labels and values
    ax.text(footer_x_start + (width + 3.6) * x_scale, footer_y_start + s(1.1),
            "SHEET NO", va='center', ha='left', fontsize=FONTSIZE, weight='bold')
    ax.text(footer_x_start + (width + 3.6) * x_scale, footer_y_start + s(0.5),
            "TOTAL SHEET", va='center', ha='left', fontsize=FONTSIZE, weight='bold')

    ax.text(footer_x_start + (width + 5.5) * x_scale, footer_y_start + s(1.1),
            str(page_num), va='center', ha='left', fontsize=FONTSIZE, weight='bold')

    # Date
    if pd.notna(df_title_row.get('date')):
        ax.text(footer_x_start + (width + 5.5) * x_scale, footer_y_start + s(0.35),
                str(df_title_row.get('date')), va='center', ha='left', fontsize=FONTSIZE)

# === Prepare Plotting ===
valid_symbols = ['capsule', 'single_fuse', 'dual_fuse', 'choke']
df_symbols = df[df['symbol'].astype(str).str.strip().str.lower().isin(valid_symbols)].reset_index(drop=True)

if 'circuit_id' not in df_symbols.columns and 'circuit_id' not in df_circuit.columns:
    raise ValueError("Excel data must contain a 'circuit_id' column in either terminal or circuit sheets")

# Keep circuit letters for internal ordering, but DO NOT use them to decide which junction comes first.
df_circuit['circuit_letter'] = df_circuit['circuit_name'].astype(str).str.extract(r'^([A-Z])')
df_circuit['letter_order'] = df_circuit['circuit_letter'].apply(lambda x: ord(x.upper()) - ord('A') if pd.notna(x) else -1)

# Get unique junction names in sheet-order (preserve first-seen order)
junction_names = pd.unique(df_circuit['junction_name'].astype(str).str.strip())

# Create an ordered list of circuit_ids by iterating junctions in that sheet order,
# and ordering circuits within a junction by letter then position
ordered_circuit_ids = []
for junction in junction_names:
    junction_mask = df_circuit['junction_name'].astype(str).str.strip() == junction
    junction_circuits = df_circuit[junction_mask].copy()
    if 'letter_order' in junction_circuits.columns and 'position' in junction_circuits.columns:
        junction_circuits = junction_circuits.sort_values(['letter_order', 'position'], na_position='last')
    elif 'position' in junction_circuits.columns:
        junction_circuits = junction_circuits.sort_values(['position'], na_position='last')
    ordered_circuit_ids.extend(junction_circuits['circuit_id'].tolist())

pin_spacing = 0.8

# Compute max_row_width for each junction to determine page size for RAILWAYPROJECT
# Modified to account for terminal symbols
junction_row_widths = {}
for junction in junction_names:
    current_x_pre = 1
    current_row_max_x_pre = 1
    current_terminal_count_pre = 0
    current_letter = None
    max_row_width = 0
    junction_mask = df_circuit['junction_name'].astype(str).str.strip() == junction
    junction_circuits = df_circuit[junction_mask].copy()
    if 'letter_order' in junction_circuits.columns and 'position' in junction_circuits.columns:
        junction_circuits = junction_circuits.sort_values(['letter_order', 'position'], na_position='last')
    elif 'position' in junction_circuits.columns:
        junction_circuits = junction_circuits.sort_values(['position'], na_position='last')
    circuit_list = junction_circuits['circuit_id'].tolist()
    for circuit_id_pre in circuit_list:
        r = df_circuit[df_circuit['circuit_id'] == circuit_id_pre]
        letter = r['circuit_letter'].iloc[0] if not r.empty and 'circuit_letter' in r.columns else ""
        if letter != current_letter and current_terminal_count_pre > 0:
            max_row_width = max(max_row_width, current_row_max_x_pre - 1)
            current_row_max_x_pre = 1
            current_x_pre = 1
            current_terminal_count_pre = 0
        current_letter = letter
        group_pre = df_symbols[df_symbols['circuit_id'] == circuit_id_pre].sort_index().reset_index(drop=True)
        total_terminals = 0
        added_width = 0
        i = 0
        while i < len(group_pre):
            symbol = str(group_pre.iloc[i].get('symbol', '')).strip().lower()
            if symbol == 'dual_fuse':
                if i + 1 < len(group_pre):
                    added_width += pin_spacing * 1.0 + pin_spacing * 1.5
                    total_terminals += 2
                    i += 2
                else:
                    added_width += pin_spacing
                    total_terminals += 1
                    i += 1
            else:
                added_width += pin_spacing
                total_terminals += 1
                i += 1
        if current_terminal_count_pre + total_terminals > 36:
            max_row_width = max(max_row_width, current_row_max_x_pre - 1)
            current_row_max_x_pre = 1
            current_x_pre = 1
            current_terminal_count_pre = 0
        current_x_pre += added_width + CIRCUIT_GAP
        current_row_max_x_pre = max(current_row_max_x_pre, current_x_pre)
        current_terminal_count_pre += total_terminals
    max_row_width = max(max_row_width, current_row_max_x_pre - 1)
    junction_row_widths[junction] = max_row_width

global_max_width = max(junction_row_widths.values()) + 2.0 if junction_row_widths else 30.0

# Compute page dimensions for JB-20(F)
jb20f_junction = 'JB-20(F)'
max_rows_visible = 3
max_terminal_symbols_per_row = 36
junction_mask = df_circuit['junction_name'].astype(str).str.strip() == jb20f_junction
junction_circuits = df_circuit[junction_mask].copy()
if 'letter_order' in junction_circuits.columns and 'position' in junction_circuits.columns:
    junction_circuits = junction_circuits.sort_values(['letter_order', 'position'], na_position='last')
elif 'position' in junction_circuits.columns:
    junction_circuits = junction_circuits.sort_values(['position'], na_position='last')
jb20f_circuit_ids = junction_circuits['circuit_id'].tolist()
num_circuits = len(jb20f_circuit_ids)
num_rows = max_rows_visible  # Since we split, but for height, assume max
fixed_fig_width = 42.8
fixed_fig_height = 31.0

bottom_margin = 1.0
top_margin = 3.0
fixed_ylim_min = CAPSULE_Y_CENTER_BASE + vertical_gap * (1 - max_rows_visible) + y_bottom_bus_offset - 1.8 - bottom_margin - footer_height
fixed_ylim_max = CAPSULE_Y_CENTER_BASE + y_top_bus_offset + 1.8 + top_margin

# Create pages grouped by junction name, preserving the junction order from the sheet.
pages = []
for junction in junction_names:
    junction_mask = df_circuit['junction_name'].astype(str).str.strip() == junction
    junction_circuits = df_circuit[junction_mask].copy()
    if 'letter_order' in junction_circuits.columns and 'position' in junction_circuits.columns:
        junction_circuits = junction_circuits.sort_values(['letter_order', 'position'], na_position='last')
    elif 'position' in junction_circuits.columns:
        junction_circuits = junction_circuits.sort_values(['position'], na_position='last')
    circuit_list = junction_circuits['circuit_id'].tolist()

    # Now, simulate drawing to split into pages
    current_page_circuits = []
    current_row_index = 0
    current_terminal_count = 0
    current_letter = None
    for cid in circuit_list:
        # Get letter
        r = df_circuit[df_circuit['circuit_id'] == cid]
        letter = r['circuit_letter'].iloc[0] if not r.empty and 'circuit_letter' in r.columns else ""

        # If new letter and not at row start, would force new row
        if letter != current_letter and current_terminal_count > 0:
            # Would force new row
            current_row_index += 1
            if current_row_index >= max_rows_visible:
                # Start new page
                if current_page_circuits:
                    pages.append((junction, current_page_circuits))
                current_page_circuits = []
                current_row_index = 0
                current_terminal_count = 0
            else:
                # Continue on same page, but reset terminal count for new row
                current_terminal_count = 0

        current_letter = letter

        # Compute terminals for this circuit
        group = df_symbols[df_symbols['circuit_id'] == cid].sort_index().reset_index(drop=True)
        total_terminals_this = 0
        i = 0
        while i < len(group):
            symbol = str(group.iloc[i].get('symbol', '')).strip().lower()
            if symbol == 'dual_fuse' and i + 1 < len(group):
                total_terminals_this += 2
                i += 2
            else:
                total_terminals_this += 1
                i += 1

        # Check if adding would exceed current row
        if current_terminal_count + total_terminals_this > max_terminal_symbols_per_row:
            # Would start new row
            current_row_index += 1
            if current_row_index >= max_rows_visible:
                # Start new page
                if current_page_circuits:
                    pages.append((junction, current_page_circuits))
                current_page_circuits = []
                current_row_index = 0
                current_terminal_count = 0
            else:
                current_terminal_count = 0

        # Add the circuit to current page
        current_page_circuits.append(cid)
        current_terminal_count += total_terminals_this

    # After all circuits in junction, add the last page if any
    if current_page_circuits:
        pages.append((junction, current_page_circuits))

total_pages = len(pages)
title_row = df_title.iloc[0] if df_title is not None and not df_title.empty else None

# Generate PDF with fixed dimensions
output_file = 'Terminal_Symbols_Centered_Fixed_Size.pdf'
with PdfPages(output_file) as pdf:
    for page_num, (junction_name, page_circuit_ids) in enumerate(pages, 1):
        # Use fixed dimensions from JB-20(F) first page
        # per-page computation
        page_max_width = 0
        current_x_page = 1
        current_row_max_x_page = 1
        current_terminal_count_page = 0
        current_letter = None
        for cid in page_circuit_ids:
            r = df_circuit[df_circuit['circuit_id'] == cid]
            letter = r['circuit_letter'].iloc[0] if not r.empty and 'circuit_letter' in r.columns else ""
            if letter != current_letter and current_terminal_count_page > 0:
                page_max_width = max(page_max_width, current_row_max_x_page - 1)
                current_row_max_x_page = 1
                current_x_page = 1
                current_terminal_count_page = 0
            current_letter = letter
            group = df_symbols[df_symbols['circuit_id'] == cid].sort_index().reset_index(drop=True)
            total_terminals = 0
            added_width = 0
            i = 0
            while i < len(group):
                symbol = str(group.iloc[i].get('symbol', '')).strip().lower()
                if symbol == 'dual_fuse':
                    if i + 1 < len(group):
                        added_width += pin_spacing * 1.0 + pin_spacing * 1.5
                        total_terminals += 2
                        i += 2
                    else:
                        added_width += pin_spacing
                        total_terminals += 1
                        i += 1
                else:
                    added_width += pin_spacing
                    total_terminals += 1
                    i += 1
            if current_terminal_count_page + total_terminals > 36:
                page_max_width = max(page_max_width, current_row_max_x_page - 1)
                current_row_max_x_page = 1
                current_x_page = 1
                current_terminal_count_page = 0
            current_x_page += added_width + CIRCUIT_GAP
            current_row_max_x_page = max(current_row_max_x_page, current_x_page)
            current_terminal_count_page += total_terminals
        page_max_width = max(page_max_width, current_row_max_x_page - 1)
        shift = (global_max_width - (page_max_width + 1.2)) / 2
        page_start_x = 1 + shift
        fig, ax = plt.subplots(figsize=(fixed_fig_width, fixed_fig_height))
        ax.set_facecolor('white')
        ax.axis('off')

        x_positions, input_connected_flags, output_connected_flags = draw_symbols(
            df_symbols, ax, page_circuit_ids, junction_name,
            start_x=page_start_x, pin_spacing=pin_spacing,
            circuits_per_page=len(page_circuit_ids),
            page_number=page_num,
            max_terminal_symbols_per_row=36,
            max_rows_visible=3,  # enforce 3 visible rows; 4th row will be blank if triggered
            page_width=global_max_width
        )

        # Draw footer on bottom right half
        left = page_start_x - 1.5  # From draw_symbols logic
        right = left + global_max_width
        draw_footer(ax, left, right, fixed_ylim_min, total_pages, page_num, title_row, junction_name)

        fig.subplots_adjust(left=0.04, right=0.99, top=0.98, bottom=0.02)
        pdf.savefig(fig, dpi=300, facecolor='white')
        plt.close(fig)
        print(f"Page {page_num} (Junction: {junction_name}) added to '{output_file}' with fixed size ({fixed_fig_width}, {fixed_fig_height})")

print(f"Multi-page PDF saved as '{output_file}'")