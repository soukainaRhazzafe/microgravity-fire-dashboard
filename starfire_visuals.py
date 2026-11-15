"""Presentation-only overrides for the StarFire app. No extra dependencies."""
import html
import streamlit as st


def apply_visual_polish():
    st.markdown("""
<style>
.stApp {background: #172337; color: #edf3fc;}
[data-testid="stHeader"] {background: transparent;}
[data-testid="stSidebar"] {background: #1d2c43;}
[data-testid="stCaptionContainer"] {color: #c0cde0;}
.lead,.sf-section-hint {color: #c3d0e3;}
.eyebrow {color: #a7eadc;}
[data-testid="stTextArea"] textarea {background: #24364e; border-color: #647b98; color: #f3f6fc;}
[data-testid="stExpander"] {background: #22334b; border-color: #516984;}
[data-testid="stVerticalBlockBorderWrapper"] > div {border-color: #4a607d !important;}
.sf-query {background: #253f4b; border-left-color: #a7eadc; color: #effbf8;}
.sf-rank {background: #304d5b; border-color: #6d98a1; color: #c1f4e9;}
.sf-area,.sf-ai-label {color: #c8d5e7;}
.sf-note {background: #283b54; border-color: #586f8c; box-shadow: 0 5px 16px #0711221c; transition: transform 180ms ease, box-shadow 180ms ease, border-color 180ms ease;}
.sf-note p {color: #f0f4fc;}
.sf-connection {background: #26484a; border-left-color: #a7eadc;}
.sf-connection .sf-note-label {color: #c1f4e6;}
.sf-evidence {background: #2c405f; border-top-color: #b0ceff;}
.sf-evidence .sf-note-label {color: #c7ddff;}
.sf-interpretation {background: #3c3658; border-top-color: #d2c2ff;}
.sf-interpretation .sf-note-label {color: #e1d7ff;}
.sf-next {background: #4a3933; border-left-color: #ffd0aa;}
.sf-next .sf-note-label {color: #ffdbc0;}
.sf-map-scroll {border-color: #536c88;}
.sf-map th {background: #30445f; color: #d6e2f2;}
.sf-map td {background: #22334b; border-top-color: #445b77;}
.sf-map a {color: #f0f5fc;}
.sf-map .sf-map-number {color: #c3d2e5;}
.sf-map tr:focus-within td {background: #354e6b;}
.sf-listed {background: #315856; border-color: #7eaaa0; color: #d4f9ed;}
.sf-absent {background: #2a3a50; border-color: #8191a7; color: #d3dce9;}
.sf-area-chip {display: inline-block; padding: .3rem .55rem; border-radius: 8px; background: #444365; border: 1px solid #817ca9; color: #e8e3ff; font-size: .76rem; line-height: 1.5; text-align: left;}
.sf-map td.sf-area-cell {text-align: left; max-width: 190px;}
.sf-map th:first-child {width: 44%;}
.sf-map th:nth-child(2) {text-align: left;}
.sf-map a:focus-visible {outline: 3px solid #ffd5b5; outline-offset: 4px; border-radius: 3px;}
.stButton > button,.stDownloadButton > button,.stLinkButton > a {background: #2c405b; border-color: #7187a3; color: #f0f5fc; transition: background 160ms ease, border-color 160ms ease;}
button[kind="primary"],button[kind="primaryFormSubmit"] {background: #ffd0aa; border-color: #ffd0aa; color: #202b3d;}
@media (hover: hover) and (pointer: fine) {
    .sf-note:hover {transform: translateY(-3px); box-shadow: 0 10px 22px #07112238; border-color: #a8bfd9;}
    .sf-map tr:hover td {background: #354e6b;}
    .stLinkButton > a:hover,.stDownloadButton > button:hover {background: #3c5676; border-color: #b0ceff; color: #fff;}
}
@keyframes sf-reveal {from {opacity: .65; transform: translateY(5px);} to {opacity: 1; transform: translateY(0);}}
.sf-note-grid {animation: sf-reveal 260ms ease-out both;}
@media (prefers-reduced-motion: reduce) {
    .sf-note-grid {animation: none !important;}
    .sf-note,.stButton > button,.stDownloadButton > button,.stLinkButton > a {transition: none !important;}
    .sf-note:hover {transform: none !important;}
}
/* Warm cream palette */
.stApp {
    background: #F7F3EC;
    color: #283548;
}

h1, h2, h3, h4, h5, h6 {
    color: #283548 !important;
}

[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stSidebar"] {
    background: #EEE8DF;
    color: #283548;
}

[data-testid="stCaptionContainer"],
.lead,
.sf-section-hint,
.sf-area,
.sf-ai-label {
    color: #596577;
}

.eyebrow {
    color: #316B62;
}

/* Search and controls */
[data-testid="stTextArea"] textarea {
    background: #FFFDFA;
    color: #283548;
    border-color: #B8B0A5;
}

[data-testid="stTextArea"] textarea::placeholder {
    color: #6B7280;
}

[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] p {
    color: #283548;
}

.stButton > button,
.stDownloadButton > button,
.stLinkButton > a {
    background: #FFFDFA;
    border-color: #B8B0A5;
    color: #283548;
}

button[kind="primary"],
button[kind="primaryFormSubmit"] {
    background: #305E56;
    border-color: #305E56;
    color: #FFFFFF;
}

/* Question and investigation labels */
.sf-query {
    background: #E5EEE7;
    border-left-color: #46786B;
    color: #283548;
}

.sf-rank {
    background: #E1EBE5;
    border-color: #A2BCAF;
    color: #305E56;
}

/* Research-note panels */
.sf-note {
    background: #FFFDFA;
    border-color: #D3CBC0;
    box-shadow: 0 4px 14px #40352608;
}

.sf-note p {
    color: #283548;
}

.sf-connection {
    background: #E5EEE7;
    border-left-color: #46786B;
}

.sf-connection .sf-note-label {
    color: #305E56;
}

.sf-evidence {
    background: #E8EEF6;
    border-top-color: #6482AC;
}

.sf-evidence .sf-note-label {
    color: #385879;
}

.sf-interpretation {
    background: #EEE8F5;
    border-top-color: #9279AF;
}

.sf-interpretation .sf-note-label {
    color: #665080;
}

.sf-next {
    background: #F8EADC;
    border-left-color: #B87B48;
}

.sf-next .sf-note-label {
    color: #80512D;
}

/* Shortlist map */
.sf-map-scroll {
    border-color: #D3CBC0;
}

.sf-map th {
    background: #EDE7DE;
    color: #465366;
}

.sf-map td {
    background: #FFFDFA;
    border-top-color: #E1DAD0;
}

.sf-map a {
    color: #305E56;
}

.sf-map .sf-map-number {
    color: #596577;
}

.sf-area-chip {
    background: #EEE8F5;
    border-color: #C9BADB;
    color: #665080;
}

.sf-listed {
    background: #E1EDE5;
    border-color: #A2BCAF;
    color: #305E56;
}

.sf-absent {
    background: #F2EEE8;
    border-color: #B8B0A5;
    color: #596577;
}

.sf-map tr:focus-within td {
    background: #EEEDE3;
}

/* Expanders and card borders */
[data-testid="stExpander"] {
    background: #FFFDFA;
    border-color: #D3CBC0;
    color: #283548;
}

[data-testid="stVerticalBlockBorderWrapper"] > div {
    border-color: #D3CBC0 !important;
}

/* Interaction states */
@media (hover: hover) and (pointer: fine) {
    .sf-note:hover {
        border-color: #A79D90;
        box-shadow: 0 10px 22px #40352612;
    }

    .sf-map tr:hover td {
        background: #EEEDE3;
    }

    .stLinkButton > a:hover,
    .stDownloadButton > button:hover {
        background: #E5EEE7;
        border-color: #46786B;
        color: #305E56;
    }
}

.sf-map a:focus-visible {
    outline-color: #665080;
}
</style>
""", unsafe_allow_html=True)


def _clean(value):
    if value is None:
        return ""
    value = str(value).strip()
    return "" if value.casefold() in {"nan", "none", "null", "<na>", "nat"} else value


def _escape(value):
    return html.escape(_clean(value), quote=True)


def render_visual_shortlist(results):
    """Show area and catalog descriptions, not a results or safety assessment."""
    rows = []
    for position, (_, row) in enumerate(results.iterrows(), start=1):
        title = _clean(row.get("title", "")) or "Untitled investigation"
        label = title if len(title) <= 105 else title[:105].rsplit(" ", 1)[0] + "…"
        area = _clean(row.get("research_area", "")).lstrip(":").strip()
        area_html = f'<span class="sf-area-chip">{_escape(area)}</span>' if area else '<span class="sf-absent">Not listed</span>'
        cells = []
        for field in ["objective", "approach"]:
            present = bool(_clean(row.get(field, "")))
            status, cls = ("Listed", "sf-listed") if present else ("Not listed", "sf-absent")
            cells.append(f'<td><span class="{cls}">{status}</span></td>')
        rows.append(f'<tr><td><span class="sf-map-number">{position:02d}</span><a href="#investigation-{position}" title="{_escape(title)}">{_escape(label)}</a></td><td class="sf-area-cell">{area_html}</td>{"".join(cells)}</tr>')
    table = '<div class="sf-map-scroll"><table class="sf-map" aria-label="Investigation areas and available catalog descriptions"><thead><tr><th scope="col">Investigation · jump to details</th><th scope="col">Research area</th><th scope="col">Objective</th><th scope="col">Approach</th></tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div>'
    st.markdown(table, unsafe_allow_html=True)
    st.caption("Listed means a catalog description is available—not that experimental files or measurements have been verified.")
