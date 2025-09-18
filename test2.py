import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Circle, Polygon, Rectangle
import numpy as np
import re
import os
from matplotlib.backends.backend_pdf import PdfPages
import math

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
def draw_junction_box(ax, x, y, junction_name, y_offset=21, rect_pad=0.2):
    text_raw = str(junction_name).strip()
    if not text_raw:
        return
    s = text_raw
    text_width = len(s) * 0.35
    text_height = 1.0
    font_size = max(20, min(35, 35 - (len(s) - 5) * 0.5))
    rect_width = text_width + rect_pad * 10
    rect_x = x - rect_width / 2
    rect_y = y - text_height / 2 + y_offset
    rect = Rectangle((rect_x, rect_y), rect_width, text_height,
                     linewidth=2, edgecolor='black', facecolor='none')
    ax.add_patch(rect)
    ax.text(x, y + y_offset, s, ha='center', va='center',
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

# === Relay output symbol ===
def draw_relay_output(ax, x_left, x_right, y=0, scale=1.0, text='RELAY', anchor_to_v_tip=False, v_offset=0.4):
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

def draw_group_top_symbol(ax, x_start, x_end, y, text='R1', scale=1.0, input_connected='N'):
    x = (x_start + x_end) / 2.0 if abs(x_end - x_start) < 0.1 else x_start
    line_extension = 0.35 * scale
    relay_gap = 0.1 * scale if str(input_connected).strip().upper() == 'Y' else 0.0
    base_y = y + relay_gap

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

    # Diagonal bar (mirrored like the bottom)
    diagonal_length = 0.21 * scale
    y_shift = -0.17 * scale
    diag_offset = -0.05 * scale
    left_adjust = -0.04 * scale
    right_adjust = 0.04 * scale
    down_shift = -0.01 * scale

    left_y = base_y + line_extension - diagonal_length - y_shift + left_adjust + diag_offset + down_shift
    right_y = base_y + line_extension - y_shift + right_adjust + diag_offset + down_shift

    ax.plot([x - diagonal_length / 2, x + diagonal_length / 2],
            [left_y, right_y],
            color='black', linewidth=1.2)

    # Text label
    text_offset = -0.1 * scale
    text_y = base_y + line_extension + 0.1 - y_shift + text_offset
    display_text = str(text)
    ax.text(x, text_y, display_text, ha='center', va='bottom',
            fontsize=int(17 * scale), fontweight='bold')

def draw_group_bottom_symbol(ax, x_start, x_end, y, text='R1', scale=1.0, output_connected='N'):
    x = (x_start + x_end) / 2.0 if abs(x_end - x_start) < 0.1 else x_start
    line_extension = 0.35 * scale
    relay_gap = 0.1 * scale if str(output_connected).strip().upper() == 'Y' else 0.0
    base_y = y - relay_gap

    # Vertical center line
    ax.plot([x, x], [base_y, base_y - line_extension], color='black', linewidth=1)

    # One /‾‾‾‾‾‾ style segment at the bottom (diagonal first, horizontal after)
    total_width = x_end - x_start
    horizontal_ratio = 0.85  # % of segment for horizontal line
    y_bottom = base_y - 0.11 * scale
    rise_height = 0.08 * scale
    y_top = y_bottom + rise_height  # upward for bottom symbol

    seg_x0 = x_start
    seg_x2 = x_end
    seg_x1 = seg_x0 + total_width * (1 - horizontal_ratio)  # end of diagonal

    # Diagonal up (/ shape) first
    ax.plot([seg_x1, seg_x0], [y_bottom, y_top], color='black', linewidth=1)

    # Horizontal line after diagonal (slightly lower)
    horizontal_offset = 0.08 * scale  # adjust this value as needed
    ax.plot([seg_x1, seg_x2], [y_top - horizontal_offset, y_top - horizontal_offset],
            color='black', linewidth=1)

    # Diagonal bar (same as before)
    diagonal_length = 0.21 * scale
    y_shift = 0.07 * scale
    diag_offset = 0.05 * scale
    left_adjust = 0.04 * scale
    right_adjust = -0.04 * scale
    diagonal_down_shift = 0.02 * scale
    left_y = base_y - line_extension - diagonal_length + y_shift + left_adjust + diag_offset - diagonal_down_shift
    right_y = base_y - line_extension + y_shift + right_adjust + diag_offset - diagonal_down_shift
    ax.plot([x - diagonal_length/2, x + diagonal_length/2],
            [left_y, right_y],
            color='black', linewidth=1.2)

    # Text label
    text_offset = 0.2 * scale
    text_y = base_y - line_extension - diagonal_length - 0.1 + y_shift + text_offset - 0.05
    display_text = str(text)
    if len(display_text) >= 3:
        words = display_text.split()
        if len(words) > 1:
            line1 = " ".join(words[:-1])
            line2 = words[-1]
            display_text = f"{line1}\n{line2}"
        else:
            display_text = display_text[:8] + "\n" + display_text[8:]
    ax.text(x, text_y, display_text, ha='center', va='top',
            fontsize=int(17 * scale), fontweight='bold', linespacing=1.2)

# === Load Excel ===
EXCEL_FILE = r'C:\Diagram\RAILWAYPROJECT.xlsx'  # use raw string to avoid escape issues
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
JUNCTION_BOX_Y_OFFSET = 2.0

# === Header Function ===
def draw_header(ax, circuit_id, header_type, x_start, x_end, text, min_symbol_bottom=None,
                first_hook_x=None, last_hook_x=None, y_top_bus_group=0, y_bottom_bus_group=0):
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
            ha = 'right'
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

def draw_s_fuse(ax, x, y_center, terminal_name):
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
    fuse_top = y_center + SYMBOL_HEIGHT / 2
    fuse_bottom = y_center - SYMBOL_HEIGHT / 2
    top_circle_radius = SYMBOL_RADIUS * 0.8
    bottom_circle_radius = SYMBOL_RADIUS * 0.8
    ax.add_patch(Circle((x, fuse_top), radius=top_circle_radius,
                        edgecolor='black', facecolor='white', linewidth=1))
    ax.add_patch(Circle((x, fuse_bottom), radius=bottom_circle_radius,
                        edgecolor='black', facecolor='white', linewidth=1))
    start = (x, fuse_top - top_circle_radius)
    end = (x, fuse_bottom + bottom_circle_radius)
    ctrl1 = (x + SYMBOL_RADIUS * 2.2, y_center + SYMBOL_HEIGHT * 0.15)
    ctrl2 = (x - SYMBOL_RADIUS * 2.2, y_center - SYMBOL_HEIGHT * 0.15)
    t = np.linspace(0, 1, 100)
    xs = (1 - t)**3 * start[0] + 3 * (1 - t)**2 * t * ctrl1[0] + 3 * (1 - t) * t**2 * ctrl2[0] + t**3 * end[0]
    ys = (1 - t)**3 * start[1] + 3 * (1 - t)**2 * t * ctrl1[1] + 3 * (1 - t) * t**2 * ctrl2[1] + t**3 * end[1]
    ax.plot(xs, ys, color='black', linewidth=1, solid_capstyle='round')
    if pd.notna(terminal_name) and str(terminal_name).strip() != '':
        term_str = str(terminal_name)
        if term_str.endswith('.0'):
            term_str = term_str[:-2]
        ax.text(x - 0.1, y_center + 0.01, term_str, ha='center', va='center', fontsize=12)
    input_left_offset = 0.005
    if pd.notna(input_left) and str(input_left).strip() != "":
        ax.text(x - input_left_offset, fuse_top + 0.18, str(input_left),
                fontsize=12, ha='right', va='bottom', rotation=90)
    input_right_offset = 0.05
    if pd.notna(input_right) and str(input_right).strip() != "":
        ax.text(x + input_right_offset, fuse_top + 0.18, str(input_right),
                fontsize=12, ha='left', va='bottom', rotation=90)
    output_left_offset = 0.005
    if pd.notna(output_left) and str(output_left).strip() != "":
        ax.text(x - output_left_offset, fuse_bottom - 0.15, str(output_left),
                fontsize=12, ha='right', va='top', rotation=90)
    output_right_offset = 0.05
    if pd.notna(output_right) and str(output_right).strip() != "":
        ax.text(x + output_right_offset, fuse_bottom - 0.18, str(output_right),
                fontsize=12, ha='left', va='top', rotation=90)
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

def draw_dual_fuse(ax, x_left, y_center, left_term, right_term):
    left_row = find_row_by_term(left_term)
    right_row = find_row_by_term(right_term)
    left_input_conn = left_row.get('input_connected', 'N') if left_row is not None else 'N'
    left_output_conn = left_row.get('output_connected', 'N') if left_row is not None else 'N'
    right_input_conn = right_row.get('input_connected', 'N') if right_row is not None else 'N'
    right_output_conn = right_row.get('output_connected', 'N') if right_row is not None else 'N'
    INNER_SPACING_MULT = 2.8
    inner_spacing = SYMBOL_WIDTH * INNER_SPACING_MULT
    x_right = x_left + inner_spacing
    def _draw_one_s(ax, x_pos, y_c, row, term_shift=0.0):
        term = None
        input_left = input_right = output_left = output_right = None
        input_conn = 'N'
        output_conn = 'N'
        if row is not None:
            term = row.get('terminal_name')
            input_left = row.get('input_left')
            input_right = row.get('input_right')
            output_left = row.get('output_left')
            output_right = row.get('output_right')
            input_conn = row.get('input_connected', 'N')
            output_conn = row.get('output_connected', 'N')
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
        if pd.notna(term):
            term_str = str(term)
            if term_str.endswith('.0'):
                term_str = term_str[:-2]
            text_x = x_pos + term_shift
            ax.text(text_x, y_c, term_str,
                    ha='center', va='center', fontsize=12)
        label_offset = 0.11
        if pd.notna(input_left):
            ax.text(x_pos - label_offset + 0.11, fuse_top + 0.24, str(input_left),
                    fontsize=12, ha='right', va='bottom', rotation=90)
        if pd.notna(input_right):
            ax.text(x_pos + 0.1, fuse_top + 0.27, str(input_right),
                    fontsize=12, ha='center', va='bottom', rotation=90)
        if pd.notna(output_left):
            ax.text(x_pos - label_offset + 0.11, fuse_bottom - 0.30, str(output_left),
                    fontsize=12, ha='right', va='top', rotation=90)
        if pd.notna(output_right):
            ax.text(x_pos + label_offset - 0.09, fuse_bottom - 0.28, str(output_right),
                    fontsize=12, ha='left', va='top', rotation=90)
        top_conn = (x_pos, fuse_top + top_circle_radius)
        bottom_conn = (x_pos, fuse_bottom - bottom_circle_radius)
        return top_conn, bottom_conn, input_conn, output_conn
    left_top, left_bottom, left_ic, left_oc = _draw_one_s(ax, x_left, y_center, left_row, term_shift=-0.1)
    right_top, right_bottom, right_ic, right_oc = _draw_one_s(ax, x_right, y_center, right_row, term_shift=-0.1)
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
            small = max(0.06, gap)
            ax.plot([x_start - small, x_start + small], [bus_y, bus_y], color='black', linewidth=1)
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

# === Main Draw Function ===
def draw_symbols(df_symbols, ax, ordered_circuit_ids, junction_name, start_x=1, pin_spacing=0.8, circuits_per_page=12, page_number=1):
    circuits_per_row = 3
    current_x = start_x
    current_circuit_count = 0
    y_offset = 0
    all_x_positions = []
    all_input_connected_flags = []
    all_output_connected_flags = []
    min_y = float('inf')
    max_y = float('-inf')
    overall_max_x = start_x
    current_row_max_x = start_x

    # Use page-specific circuit ids supplied (ordered already per-junction)
    page_circuit_ids = ordered_circuit_ids

    for circuit_id in page_circuit_ids:
        circuit_rows = df_circuit[df_circuit['circuit_id'] == circuit_id]
        circuit_pos = circuit_rows['position'].iloc[0] if not circuit_rows.empty and 'position' in circuit_rows.columns else None
        group = df_symbols[df_symbols['circuit_id'] == circuit_id].sort_index().reset_index(drop=True)
        circuit_name_row = circuit_rows
        circuit_name = get_block_circuit_name(circuit_name_row)

        if current_circuit_count >= circuits_per_row:
            overall_max_x = max(overall_max_x, current_row_max_x)
            current_row_max_x = start_x
            y_offset -= vertical_gap
            current_x = start_x
            current_circuit_count = 0

        capsule_y_center = CAPSULE_Y_CENTER_BASE + y_offset
        y_top_bus_group = capsule_y_center + y_top_bus_offset
        y_bottom_bus_group = capsule_y_center + y_bottom_bus_offset

        if current_circuit_count == 0:
            draw_circuit_name(ax, start_x - 1.2, capsule_y_center, circuit_name or f"Circuit {circuit_id}")

        if group.empty:
            current_x += pin_spacing + CIRCUIT_GAP
            current_row_max_x = max(current_row_max_x, current_x)
            current_circuit_count += 1
            min_y = min(min_y, capsule_y_center - 2.0)
            max_y = max(max_y, capsule_y_center + 2.0)
            continue

        input_connected_flags = []
        output_connected_flags = []
        x_positions = []
        symbol_bottoms = []
        terminal_names_for_positions = []

        i = 0
        while i < len(group):
            row = group.iloc[i]
            symbol = str(row.get('symbol', '')).strip().lower()
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
                i += 1
            elif symbol == 'single_fuse':
                top_conn, bottom_conn, input_conn, output_conn = draw_s_fuse(
                    ax, current_x, capsule_y_center, row.get('terminal_name')
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
                i += 1
            elif symbol == 'choke':
                top_conn, bottom_conn, input_conn, output_conn = draw_choke(
                    ax, current_x, capsule_y_center, row.get('terminal_name')
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
                i += 1
            elif symbol == 'dual_fuse':
                if i + 1 < len(group):
                    next_row = group.iloc[i+1]
                    LEFT_EXTRA = pin_spacing * 1.0
                    AFTER_SPACING = pin_spacing * 1.5
                    current_x += LEFT_EXTRA
                    dual_start_x = current_x - SYMBOL_WIDTH * 1.25
                    top_conn, bottom_conn, left_ic, left_oc, right_ic, right_oc = draw_dual_fuse(
                        ax, dual_start_x, capsule_y_center,
                        row.get('terminal_name'),
                        next_row.get('terminal_name')
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
                    i += 2
                else:
                    top_conn, bottom_conn, input_conn, output_conn = draw_s_fuse(
                        ax, current_x, capsule_y_center, row.get('terminal_name')
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
                    i += 1
            else:
                i += 1

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
            start_idx = terminal_names_for_positions.index(start_name)
            end_idx = terminal_names_for_positions.index(end_name)
            if start_idx > end_idx:
                start_idx, end_idx = end_idx, start_idx
            x_left = x_positions[start_idx]
            x_right = x_positions[end_idx]

            if header_type == 'RELAY':
                if input_output == 'input':
                    symbol_top_y = capsule_y_center + SYMBOL_HEIGHT/2 + SYMBOL_RADIUS
                    input_conn_flag = any(name_to_input_connected.get(str(term).strip().replace('.0', ''), False) for term in [terminal_start, terminal_end])
                    vertical_line_start = y_top_bus_group if input_conn_flag else symbol_top_y + stub_length
                    draw_group_top_symbol(ax, x_left, x_right, vertical_line_start, text=text, input_connected='Y' if input_conn_flag else 'N')
                elif input_output == 'output':
                    symbol_bottom_y = capsule_y_center - SYMBOL_HEIGHT/2 - SYMBOL_RADIUS
                    output_conn_flag = any(name_to_output_connected.get(str(term).strip().replace('.0', ''), False) for term in [terminal_start, terminal_end])
                    vertical_line_end = y_bottom_bus_group if output_conn_flag else symbol_bottom_y - stub_length
                    draw_group_bottom_symbol(ax, x_left, x_right, vertical_line_end, text=text, output_connected='Y' if output_conn_flag else 'N')
            elif header_type in ['WIREFROM', 'WIRETO']:
                min_symbol_bottom_local = min(symbol_bottoms[start_idx:end_idx+1]) if symbol_bottoms else None
                if header_type == 'WIREFROM':
                    sub_flags = input_connected_flags[start_idx:end_idx+1]
                    connected_local = [i for i, f in enumerate(sub_flags) if f]
                    first_hook_x_specific = x_positions[start_idx + connected_local[0]] if connected_local else x_left
                    draw_header(ax, circuit_id, header_type, x_left, x_right, text,
                                min_symbol_bottom=min_symbol_bottom_local,
                                first_hook_x=first_hook_x_specific,
                                last_hook_x=None,
                                y_top_bus_group=y_top_bus_group,
                                y_bottom_bus_group=y_bottom_bus_group)
                elif header_type == 'WIRETO':
                    sub_flags = output_connected_flags[start_idx:end_idx+1]
                    connected_local = [i for i, f in enumerate(sub_flags) if f]
                    last_hook_x_specific = x_positions[start_idx + connected_local[-1]] if connected_local else x_right
                    draw_header(ax, circuit_id, header_type, x_left, x_right, text,
                                min_symbol_bottom=min_symbol_bottom_local,
                                first_hook_x=None,
                                last_hook_x=last_hook_x_specific,
                                y_top_bus_group=y_top_bus_group,
                                y_bottom_bus_group=y_bottom_bus_group)

        all_x_positions.extend(x_positions)
        all_input_connected_flags.extend(input_connected_flags)
        all_output_connected_flags.extend(output_connected_flags)
        current_x += CIRCUIT_GAP
        current_row_max_x = max(current_row_max_x, current_x)
        current_circuit_count += 1
        min_y = min(min_y, y_bottom_bus_group - 1.8)
        max_y = max(max_y, y_top_bus_group + 1.8)

    overall_max_x = max(overall_max_x, current_row_max_x)
    margin = 1.0
    if all_x_positions:
        ax.set_xlim(start_x - 1.5, overall_max_x - 0.5)
    else:
        ax.set_xlim(0, 10)
    if min_y == float('inf') or max_y == float('-inf'):
        ax.set_ylim(0, 5)
    else:
        ax.set_ylim(min_y, max_y + JUNCTION_BOX_Y_OFFSET)

    if all_x_positions:
        page_center_x = (min(all_x_positions) + max(all_x_positions)) / 2
    else:
        page_center_x = (start_x + overall_max_x) / 2
    junction_box_y = CAPSULE_Y_CENTER_BASE + JUNCTION_BOX_Y_OFFSET + y_offset
    draw_junction_box(ax, page_center_x, junction_box_y, junction_name)

    return all_x_positions, all_input_connected_flags, all_output_connected_flags

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
# and ordering circuits within a junction by letter then start_no
ordered_circuit_ids = []
for junction in junction_names:
    junction_mask = df_circuit['junction_name'].astype(str).str.strip() == junction
    junction_circuits = df_circuit[junction_mask].copy()
    if 'letter_order' in junction_circuits.columns and 'start_no' in junction_circuits.columns:
        junction_circuits = junction_circuits.sort_values(['letter_order', 'start_no'], na_position='last')
    elif 'start_no' in junction_circuits.columns:
        junction_circuits = junction_circuits.sort_values(['start_no'], na_position='last')
    ordered_circuit_ids.extend(junction_circuits['circuit_id'].tolist())

circuits_per_row = 3
pin_spacing = 0.8

# Compute max_row_width for each junction to determine page size for JB-20(F)
junction_row_widths = {}
for junction in junction_names:
    current_x_pre = 1
    current_row_max_x_pre = 1
    max_row_width = 0
    current_circuit_count_pre = 0
    junction_mask = df_circuit['junction_name'].astype(str).str.strip() == junction
    junction_circuits = df_circuit[junction_mask].copy()
    if 'letter_order' in junction_circuits.columns and 'start_no' in junction_circuits.columns:
        junction_circuits = junction_circuits.sort_values(['letter_order', 'start_no'], na_position='last')
    elif 'start_no' in junction_circuits.columns:
        junction_circuits = junction_circuits.sort_values(['start_no'], na_position='last')
    circuit_list = junction_circuits['circuit_id'].tolist()

    for circuit_id_pre in circuit_list:
        group_pre = df_symbols[df_symbols['circuit_id'] == circuit_id_pre].sort_index().reset_index(drop=True)
        slot_vals = []
        for idx, row in group_pre.iterrows():
            slot = None
            for colname in ['position', 'pos', 'slot', 'terminal_position', 'terminal_no', 'start_no']:
                if colname in group_pre.columns:
                    val = row.get(colname)
                    if pd.notna(val):
                        try:
                            slot = int(float(val))
                            break
                        except Exception:
                            pass
            if slot is None:
                tn = row.get('terminal_name')
                try:
                    if pd.notna(tn):
                        tni = int(float(str(tn).strip()))
                        slot = tni
                except Exception:
                    slot = None
            slot_vals.append(slot)
        total_slots = len(group_pre) if all(s is None for s in slot_vals) else (max([s for s in slot_vals if s is not None]) - min([s for s in slot_vals if s is not None]) + 1) if any(s is not None for s in slot_vals) else len(group_pre) if len(group_pre) > 0 else 1

        if current_circuit_count_pre >= circuits_per_row:
            max_row_width = max(max_row_width, current_row_max_x_pre - 1)
            current_row_max_x_pre = 1
            current_x_pre = 1
            current_circuit_count_pre = 0

        current_x_pre += total_slots * pin_spacing + CIRCUIT_GAP
        current_row_max_x_pre = max(current_row_max_x_pre, current_x_pre)
        current_circuit_count_pre += 1

    max_row_width = max(max_row_width, current_row_max_x_pre - 1)
    junction_row_widths[junction] = max_row_width

# Compute page dimensions for JB-20(F)
jb20f_junction = 'JB-20(F)'
circuits_per_page = 12
junction_mask = df_circuit['junction_name'].astype(str).str.strip() == jb20f_junction
junction_circuits = df_circuit[junction_mask].copy()
if 'letter_order' in junction_circuits.columns and 'start_no' in junction_circuits.columns:
    junction_circuits = junction_circuits.sort_values(['letter_order', 'start_no'], na_position='last')
elif 'start_no' in junction_circuits.columns:
    junction_circuits = junction_circuits.sort_values(['start_no'], na_position='last')
jb20f_circuit_ids = junction_circuits['circuit_id'].tolist()
jb20f_pages = []
for i in range(0, len(jb20f_circuit_ids), circuits_per_page):
    chunk = jb20f_circuit_ids[i:i+circuits_per_page]
    if chunk:
        jb20f_pages.append((jb20f_junction, chunk))

# Use the first page of JB-20(F) to determine dimensions
num_circuits = len(jb20f_pages[0][1]) if jb20f_pages else 0
num_rows = ((num_circuits - 1) // circuits_per_row) + 1 if num_circuits > 0 else 1
fixed_fig_width = max(10, junction_row_widths.get(jb20f_junction, 10) + 1.0)
fixed_fig_height = max(4, num_rows * 6.0 * 2)

# Create pages grouped by junction name, preserving the junction order from the sheet.
pages = []
for junction in junction_names:
    junction_mask = df_circuit['junction_name'].astype(str).str.strip() == junction
    junction_circuits = df_circuit[junction_mask].copy()
    if 'letter_order' in junction_circuits.columns and 'start_no' in junction_circuits.columns:
        junction_circuits = junction_circuits.sort_values(['letter_order', 'start_no'], na_position='last')
    elif 'start_no' in junction_circuits.columns:
        junction_circuits = junction_circuits.sort_values(['start_no'], na_position='last')
    circuit_list = junction_circuits['circuit_id'].tolist()

    # Split circuit_list into chunks of circuits_per_page
    for i in range(0, len(circuit_list), circuits_per_page):
        chunk = circuit_list[i:i+circuits_per_page]
        if chunk:
            pages.append((junction, chunk))

# Generate PDF with fixed dimensions
output_file = 'Terminal_Symbols_Centered_Fixed_Size.pdf'
with PdfPages(output_file) as pdf:
    for page_num, (junction_name, page_circuit_ids) in enumerate(pages, 1):
        # Use fixed dimensions from JB-20(F) first page
        fig, ax = plt.subplots(figsize=(fixed_fig_width, fixed_fig_height))
        ax.set_facecolor('white')
        ax.axis('off')

        x_positions, input_connected_flags, output_connected_flags = draw_symbols(
            df_symbols, ax, page_circuit_ids, junction_name,
            start_x=1, pin_spacing=pin_spacing,
            circuits_per_page=circuits_per_page,
            page_number=page_num
        )

        plt.tight_layout()
        pdf.savefig(fig, dpi=300, facecolor='white')
        plt.close(fig)
        print(f"Page {page_num} (Junction: {junction_name}) added to '{output_file}' with fixed size ({fixed_fig_width}, {fixed_fig_height})")

print(f"Multi-page PDF saved as '{output_file}'")
