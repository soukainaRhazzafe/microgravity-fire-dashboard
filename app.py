import streamlit as st

st.set_page_config(
    page_title="Microgravity Dashboard",
    layout="wide",
)

INVESTIGATIONS = [
    {
        "name": "BASS",
        "full_name": "Burning and Suppression of Solids",
        "categories": ["Prevention", "Suppression"],
        "fuel_type": "Solid materials",
        "summary": (
            "Investigates solid-material burning and suppression in "
            "microgravity, with relevance to spacecraft materials "
            "and fire safety."
        ),
        "source_url": (
            "https://www.nasa.gov/glenn/"
            "glenn-expertise-space-exploration/"
            "physical-sciences-program/combustion-science/"
        ),
    },
    {
        "name": "SAME",
        "full_name": "Smoke and Aerosol Measurement Experiment",
        "categories": ["Detection"],
        "fuel_type": "Smoke and aerosols from material samples",
        "summary": (
            "Studies smoke and aerosol measurements in microgravity "
            "to support understanding of spacecraft fire detection."
        ),
        "source_url": (
            "https://www.nasa.gov/glenn/"
            "glenn-expertise-space-exploration/"
            "physical-sciences-program/combustion-science/"
        ),
    },
    {
        "name": "FLEX",
        "full_name": "Flame Extinguishment Experiment",
        "categories": ["Suppression"],
        "fuel_type": "Liquid-fuel droplets",
        "summary": (
            "Studies droplet combustion and flame extinction in "
            "microgravity, including suppression through atmospheric "
            "dilution with carbon dioxide and helium."
        ),
        "source_url": (
            "https://www.nasa.gov/glenn/"
            "glenn-expertise-space-exploration/"
            "physical-sciences-program/combustion-science/"
        ),
    },
]

st.title("Microgravity Fire Dashboard")
st.write(
    "Explore NASA combustion investigations by fire-safety research area."
)

st.caption(
    "Prototype catalog: manually curated from the supplied challenge brief. "
    "These are introductory summaries, not experimental findings or "
    "validated safety recommendations."
)

st.sidebar.header("Explore investigations")

selected_category = st.sidebar.selectbox(
    "Research area",
    ["All", "Prevention", "Detection", "Suppression"],
)

search_text = st.sidebar.text_input(
    "Search investigations",
    placeholder="For example: droplets",
)

filtered_investigations = []

for investigation in INVESTIGATIONS:
    matches_category = (
        selected_category == "All"
        or selected_category in investigation["categories"]
    )

    searchable_text = " ".join(
        [
            investigation["name"],
            investigation["full_name"],
            investigation["fuel_type"],
            investigation["summary"],
        ]
    ).lower()

    matches_search = search_text.strip().lower() in searchable_text

    if matches_category and matches_search:
        filtered_investigations.append(investigation)

st.subheader("Investigation catalog")
st.write(f"Showing {len(filtered_investigations)} of {len(INVESTIGATIONS)} investigations.")

if not filtered_investigations:
    st.info("No matching investigations. Try another category or search term.")

for investigation in filtered_investigations:
    with st.container(border=True):
        st.subheader(
            f"{investigation['name']} — {investigation['full_name']}"
        )
        st.write(investigation["summary"])

        st.write(
            "Research areas: "
            + ", ".join(investigation["categories"])
        )
        st.write(f"Fuel / material context: {investigation['fuel_type']}")

        with st.expander("Sources and limitations"):
            st.write(
                "Summary provenance: manually curated from the "
                "NASA Space Apps challenge brief supplied for this project."
            )
            st.link_button(
                "NASA combustion research overview",
                investigation["source_url"],
            )
            st.write(
                "This link provides background information, not a "
                "specific experimental run or measurement. Run-level "
                "conditions and PSI dataset links have not yet been added."
            )
