import streamlit as st

from recommender import (
    PSIInvestigationRecommender
)


st.set_page_config(
    page_title="NASA PSI Research Assistant",
    page_icon="🔥",
    layout="wide"
)


@st.cache_resource
def load_recommender():

    return PSIInvestigationRecommender(
        "investigations.csv"
    )


recommender = load_recommender()


st.title(
    "🔥 NASA PSI Fire Safety Research Assistant"
)

st.markdown(
    """
Find the NASA PSI combustion investigations
most relevant to your research question.
    """
)


query = st.text_area(
    "Research Goal",
    placeholder="""
Examples:

Predict fire spread in a lunar habitat.

Study smoke detector performance
inside a pressurized rover.

Investigate fire suppression
using carbon dioxide.
"""
)


top_k = st.slider(
    "Number of Results",
    min_value=3,
    max_value=10,
    value=5
)


if st.button("Recommend"):

    if not query.strip():

        st.warning(
            "Please enter a research goal."
        )

    else:

        results = recommender.recommend(
            query,
            top_k
        )

        st.subheader(
            "Recommended Investigations"
        )

        for _, row in results.iterrows():

            with st.expander(
                f"{row['title']}  |  Score: {row['score']:.3f}",
                expanded=True
                ):

                st.write(
                    f"**Research Area:** {row['research_area'].astype(str).str.replace(r"^\s*:\s*", "", regex=True).str.strip()}"
                )

                st.write(
                    f"**Hardware:** {row['hardware']}"
                )

                st.write(
                    f"**Objective:** {row['objective']}"
                )

                st.write(
                    f"**Approach:** {row['approach']}"
                )

                st.write(
                    f"**Hypothesis:** {row['hypothesis']}"
                )

                st.write(
                    f"**Research Impacts:** {row['research_impacts']}"
                )

                st.markdown(
                    f"{row['source_url']}"
                )

        st.subheader(
            "Results Table"
        )

        st.dataframe(
            results[
                [
                    "title",
                    "research_area",
                    "objective",
                    "score"
                ]
            ],
            use_container_width=True,
        )

        csv = results.to_csv(
            index=False
        )

        st.download_button(
            label="Download Results",
            data=csv,
            file_name="recommended_investigations.csv",
            mime="text/csv",
        )