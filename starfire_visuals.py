"""StarFire cream theme with interactive cards. Presentation only."""
import html
import streamlit as st


def apply_visual_polish():
    st.markdown("""
<style>
.stApp {background: #F7F3EC; color: #283548;}
[data-testid="stHeader"] {background: transparent;}
h1,h2,h3,h4,h5,h6 {color: #283548 !important;}
[data-testid="stSidebar"] {background: #EEE8DF; color: #283548;}
[data-testid="stSidebar"] p,[data-testid="stSidebar"] label {color: #283548;}
[data-testid="stCaptionContainer"],.lead,.sf-section-hint,.sf-area,.sf-ai-label {color: #596577;}
.eyebrow {color: #316B62;}
[data-testid="stTextArea"] textarea {background: #FFFDFA; color: #283548; border-color: #B8B0A5;}
[data-testid="stTextArea"] textarea::placeholder {color: #6B7280;}
[data-testid="stWidgetLabel"],[data-testid="stWidgetLabel"] p {color: #283548;}
[data-testid="stExpander"] {background: #FFFDFA; border-color: #D3CBC0; color: #283548;}
[data-testid="stExpander"] summary,[data-testid="stExpander"] summary p {color: #283548;}
[data-testid="stExpander"] [data-testid="stMarkdownContainer"] {color: #283548;}
[data-testid="stVerticalBlockBorderWrapper"] > div {border-color: #D3CBC0 !important;}
.sf-query {background: #E5EEE7; border-left-color: #46786B; color: #283548;}
.sf-rank {background: #E1EBE5; border-color: #A2BCAF; color: #305E56;}
.sf-note {background: #FFFDFA; border-color: #D3CBC0; box-shadow: 0 4px 14px #40352608; transition: transform 180ms ease,box-shadow 180ms ease,border-color 180ms ease;}
.sf-note p {color: #283548;}
.sf-connection {background: #E5EEE7; border-left-color: #46786B;}
.sf-connection .sf-note-label {color: #305E56;}
.sf-evidence {background: #E8EEF6; border-top-color: #6482AC;}
.sf-evidence .sf-note-label {color: #385879;}
.sf-interpretation {background: #EEE8F5; border-top-color: #9279AF;}
.sf-interpretation .sf-note-label {color: #665080;}
.sf-next {background: #F8EADC; border-left-color: #B87B48;}
.sf-next .sf-note-label {color: #80512D;}
.sf-map-scroll {border-color: #D3CBC0;}
.sf-map th {background: #EDE7DE; color: #465366;}
.sf-map td {background: #FFFDFA; border-top-color: #E1DAD0;}
.sf-map a {color: #305E56;}
.sf-map .sf-map-number {color: #596577;}
.sf-area-chip {display: inline-block; padding: .3rem .55rem; border-radius: 8px; background: #EEE8F5; border: 1px solid #C9BADB; color: #665080; font-size: .76rem; line-height: 1.5; text-align: left;}
.sf-map td.sf-area-cell {text-align: left; max-width: 190px;}
.sf-map th:first-child {width: 44%;}
.sf-map th:nth-child(2) {text-align: left;}
.sf-listed {background: #E1EDE5; border-color: #A2BCAF; color: #305E56;}
.sf-absent {background: #F2EEE8; border-color: #B8B0A5; color: #596577;}
.sf-map tr:focus-within td {background: #EEEDE3;}
.sf-map a:focus-visible {outline: 3px solid #665080; outline-offset: 4px; border-radius: 3px;}
.stButton > button,.stDownloadButton > button,.stLinkButton > a {background: #FFFDFA; border-color: #B8B0A5; color: #283548; transition: background 160ms ease,border-color 160ms ease;}
button[kind="primary"],button[kind="primaryFormSubmit"] {background: #305E56; border-color: #305E56; color: #FFFFFF;}
[id^="investigation-"] {position: relative;}
[id^="investigation-"]:target::after {content: "Selected investigation"; display: inline-block; margin: 0 0 .6rem; padding: .3rem .65rem; border: 1px solid #A2BCAF; border-radius: 7px; background: #E1EBE5; color: #305E56; font-size: .75rem;}
.sf-note-grid {animation: none;}
@keyframes sf-panel-enter {from {opacity: .65; transform: translateY(6px);} to {opacity: 1; transform: translateY(0);}}
@media (prefers-reduced-motion: no-preference) {
    .sf-note {animation: sf-panel-enter 280ms ease-out backwards;}
    .sf-connection {animation-delay: 0ms;}
    .sf-evidence {animation-delay: 70ms;}
    .sf-interpretation {animation-delay: 140ms;}
    .sf-next {animation-delay: 210ms;}
}
@media (hover: hover) and (pointer: fine) {
    .sf-note:hover {transform: translateY(-3px); border-color: #A79D90; box-shadow: 0 10px 22px #40352612;}
    .sf-map tr:hover td {background: #EEEDE3;}
    .stLinkButton > a:hover,.stDownloadButton > button:hover {background: #E5EEE7; border-color: #46786B; color: #305E56;}
}
@media (prefers-reduced-motion: no-preference) and (hover: hover) and (pointer: fine) {
    .stButton > button:active,.stDownloadButton > button:active,.stLinkButton > a:active {transform: translateY(1px);}
}
@media (prefers-reduced-motion: reduce) {
    .sf-note,.sf-note-grid {animation: none !important; transition: none !important;}
    .sf-note:hover {transform: none !important;}
    .stButton > button,.stDownloadButton > button,.stLinkButton > a {transition: none !important;}
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
