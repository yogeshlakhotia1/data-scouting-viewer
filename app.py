"""
Streamlit dashboard for exploring publicly available oncology datasets.

This app provides an interactive interface for scientists to browse and
filter curated oncology datasets drawn from the public domain.  The
data are represented as a small catalogue of Gene Expression Omnibus (GEO)
studies covering different treatment modalities—including immune
checkpoint inhibition, targeted therapy, chemotherapy and radiotherapy.
Users can filter on a number of attributes (e.g. treatment category,
treatment name, disease, tissue type, number of samples and the
availability of matched pre/post‑treatment samples) and inspect a
tabular overview of matching datasets.  Selecting a dataset opens a
detailed view with the full set of metadata, including experimental
design and links to the original source.

To run this app locally install streamlit (``pip install streamlit``)
and then execute ``streamlit run app.py``.  The app relies only on
standard Python libraries (pandas, urllib, etc.) and does not require
any external APIs.
"""

from __future__ import annotations

import urllib.parse
import pandas as pd
import streamlit as st


def load_dataset() -> pd.DataFrame:
    """Return a DataFrame describing curated oncology datasets.

    The schema of the returned DataFrame includes columns such as
    ``id``, ``title``, ``summary``, ``experimental_design``,
    ``data_modality``, ``disease``, ``tissue``, ``sampling_site``,
    ``treatment_category``, ``treatment_name``, ``treatment_info``,
    ``treatment_target``, ``total_samples``, ``total_patients``,
    ``sampling_timepoint``, ``patient_distribution``,
    ``matched_pre_post``, ``overall_survival_info``,
    ``clinical_trial_nct_ids``, ``trial_phase``,
    ``raw_data_availability``, ``processed_data_availability``,
    ``publication_availability``, ``publication_link``, ``pubmed_id``,
    ``year``, ``authors``, ``author_affiliation``, ``source``,
    ``source_link`` and ``study_cohorts``.  Additional fields may
    easily be added.
    """

    # Define a list of dictionaries, each representing a single dataset.
    datasets = [
        {
            "id": "GSE67501",
            "priority": "High",
            "title": "Renal cell carcinoma anti‑PD‑1 response dataset (GSE67501)",
            "summary": (
                "Pre‑treatment tumour expression of PD‑L1 correlates with favourable outcomes following PD‑1 blockade, "
                "but many PD‑L1‑positive tumours do not respond. This study profiled gene expression in pre‑treatment "
                "renal cell carcinoma (RCC) samples from patients treated with nivolumab to identify gene signatures "
                "associated with response or resistance【634008455624386†screenshot】.  Eleven FFPE specimens were analysed: "
                "four were from responders and seven from non‑responders."),
            "experimental_design": (
                "Gene expression profiling was performed on total RNA extracted from 11 formalin‑fixed paraffin‑embedded "
                "renal tumours. Four patients experienced an objective response to anti‑PD‑1 therapy, while seven did not. "
                "Only pre‑treatment samples were collected, and the aim was to identify predictive gene signatures."),
            "data_modality": "Expression profiling by array",
            "disease": "Renal cell carcinoma",
            "tissue": "FFPE renal tumour",
            "sampling_site": "Tumour",
            "treatment_category": "ICI",
            "treatment_name": "Nivolumab",
            "treatment_info": "Anti‑PD‑1 immune checkpoint inhibitor",
            "treatment_target": "PD‑1",
            "total_samples": 11,
            "total_patients": 11,
            "sampling_timepoint": "Pre‑treatment",
            "patient_distribution": "Responders: 4; Non‑responders: 7",
            "matched_pre_post": "No",
            "overall_survival_info": "No",
            "clinical_trial_nct_ids": "",
            "trial_phase": "",
            "raw_data_availability": "Yes (GEO)",
            "processed_data_availability": "Yes",
            "publication_availability": "Yes",
            "publication_link": "https://pubmed.ncbi.nlm.nih.gov/27491898",
            "pubmed_id": "27491898",
            "year": 2016,
            "authors": "Ascierto ML, McMiller TL et al.",
            "author_affiliation": "Johns Hopkins University",
            "source": "GEO",
            "source_link": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE67501",
            "study_cohorts": "11 RCC patients positive for PD‑L1 treated with anti‑PD‑1 therapy"
        },
        {
            "id": "GSE195832",
            "priority": "Medium",
            "title": (
                "Immunostimulatory CAF subpopulations predict immunotherapy response in head and neck cancer (GSE195832)"
            ),
            "summary": (
                "Cancer‑associated fibroblasts (CAF) can influence checkpoint immunotherapy.  In this neoadjuvant trial, "
                "single‑cell RNA sequencing and bulk RNA sequencing were performed on head and neck squamous cell carcinoma "
                "(HNSCC) tumours collected before and after nivolumab treatment.  CAF protein activity profiles derived "
                "from the paired single‑cell data were used to analyse a 28‑patient bulk cohort.  The study identified distinct "
                "CAF subsets associated with treatment response and resistance【406685514032338†L28-L55】."),
            "experimental_design": (
                "High‑dimensional single‑cell RNA sequencing (scRNA‑Seq) and bulk RNA sequencing were used. Four patients had paired "
                "scRNA‑Seq samples collected pre‑ and post‑nivolumab. The derived fibroblast protein activity profiles were then "
                "applied to a clinically annotated bulk RNA‑Seq cohort of 28 patients【406685514032338†L37-L59】."),
            "data_modality": "scRNA‑Seq and bulk RNA‑Seq",
            "disease": "Head and neck squamous cell carcinoma",
            "tissue": "Tumour",
            "sampling_site": "Tumour",
            "treatment_category": "ICI",
            "treatment_name": "Nivolumab",
            "treatment_info": "Anti‑PD‑1 therapy (immunotherapy)",
            "treatment_target": "PD‑1",
            "total_samples": 86,
            "total_patients": 32,
            "sampling_timepoint": "Pre‑ and post‑treatment",
            "patient_distribution": "",
            "matched_pre_post": "Yes",
            "overall_survival_info": "No",
            "clinical_trial_nct_ids": "",
            "trial_phase": "",
            "raw_data_availability": "Yes (GEO)",
            "processed_data_availability": "Yes",
            "publication_availability": "Yes",
            "publication_link": "https://pubmed.ncbi.nlm.nih.gov/35262677",
            "pubmed_id": "35262677",
            "year": 2022,
            "authors": "Obradovic AZ, Graves DK, Korrer MJ et al.",
            "author_affiliation": "Vanderbilt University Medical Center",
            "source": "GEO",
            "source_link": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE195832",
            "study_cohorts": "Neoadjuvant nivolumab trial in HNSCC (four patients for scRNA‑Seq; 28‑patient bulk cohort)"
        },
        {
            "id": "GSE50509",
            "priority": "High",
            "title": (
                "BRAF‑inhibitor resistance mechanisms in metastatic melanoma (GSE50509)"
            ),
            "summary": (
                "Resistance to BRAF inhibitors occurs frequently in metastatic melanoma, yet the prevalence and clinical "
                "correlates of resistance mechanisms are not fully understood.  This study analysed progressing BRAF^V600 mutant "
                "metastases from patients treated with dabrafenib or vemurafenib, identifying resistance mechanisms such as BRAF "
                "splice variants, N‑RAS mutations, BRAF amplification, MEK1/2 mutations and an AKT1 mutation.  Some progressing "
                "tumours lacked identifiable mechanisms and retained MAPK pathway inhibition, and patients with these tumours had "
                "poor outcomes【875921504785984†L27-L56】."),
            "experimental_design": (
                "Total RNA was extracted from fresh‑frozen melanoma tumours obtained before BRAF inhibitor therapy and at the time "
                "of tumour progression.  Gene expression profiling was performed using Illumina arrays to compare pre‑treatment and "
                "progression samples【875921504785984†L58-L60】."),
            "data_modality": "Expression profiling by array",
            "disease": "Metastatic melanoma",
            "tissue": "Fresh‑frozen tumour",
            "sampling_site": "Tumour",
            "treatment_category": "Targeted therapy",
            "treatment_name": "Dabrafenib or vemurafenib",
            "treatment_info": "BRAF kinase inhibitors for BRAF^V600 mutant melanoma",
            "treatment_target": "BRAF V600",
            "total_samples": 61,
            "total_patients": 30,
            "sampling_timepoint": "Pre‑treatment and progression",
            "patient_distribution": "",
            "matched_pre_post": "Yes",
            "overall_survival_info": "Yes",
            "clinical_trial_nct_ids": "",
            "trial_phase": "",
            "raw_data_availability": "Yes (GEO)",
            "processed_data_availability": "Yes",
            "publication_availability": "Yes",
            "publication_link": "https://pubmed.ncbi.nlm.nih.gov/24463458",
            "pubmed_id": "24463458",
            "year": 2014,
            "authors": "Rizos H, Pupo G et al.",
            "author_affiliation": "University of Sydney",
            "source": "GEO",
            "source_link": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE50509",
            "study_cohorts": "Metastatic melanoma patients treated with BRAF inhibitors"
        },
        {
            "id": "GSE192341",
            "priority": "Medium",
            "title": (
                "Pre‑treatment breast cancer biopsies to study neoadjuvant chemotherapy response (GSE192341)"
            ),
            "summary": (
                "Locally advanced breast cancer patients receiving neoadjuvant chemotherapy often relapse if they do not achieve "
                "a complete pathologic response.  This dataset comprises RNA‑seq profiles from 317 HER2‑negative treatment‑naïve "
                "breast cancer biopsies and deep sequencing of 22 matched pre‑ and post‑treatment tumours.  The study found that "
                "triple‑negative tumours with high proliferation and immune response and low extracellular matrix expression had "
                "better response and survival, whereas ER‑positive tumours exhibited the opposite pattern.  Paired pre/post analysis "
                "revealed numerous genomic and transcriptomic differences between pre‑ and post‑chemotherapy samples【164409062437187†L33-L56】."),
            "experimental_design": (
                "Triple‑negative and luminal breast cancer patients scheduled for neoadjuvant chemotherapy provided fresh‑frozen biopsies. "
                "RNA was isolated and sequenced using an Illumina HiSeq 2000.  Data from this cohort were analysed together with the "
                "public dataset GSE34138, yielding 317 gene expression profiles【164409062437187†L59-L63】."),
            "data_modality": "RNA‑seq and whole‑exome sequencing",
            "disease": "Breast cancer (HER2‑negative)",
            "tissue": "Fresh‑frozen breast tumour biopsy",
            "sampling_site": "Tumour",
            "treatment_category": "Chemotherapy",
            "treatment_name": "Neoadjuvant chemotherapy",
            "treatment_info": "Anthracycline‑ and taxane‑based chemotherapy given before surgery",
            "treatment_target": "Cytotoxic agents",
            "total_samples": 131,
            "total_patients": 109,
            "sampling_timepoint": "Pre‑ and post‑treatment",
            "patient_distribution": "Triple‑negative vs ER‑positive cohorts",
            "matched_pre_post": "Yes",
            "overall_survival_info": "Yes",
            "clinical_trial_nct_ids": "",
            "trial_phase": "",
            "raw_data_availability": "Yes (EGA deposit)【164409062437187†L59-L65】",
            "processed_data_availability": "Yes",
            "publication_availability": "Yes",
            "publication_link": "https://pubmed.ncbi.nlm.nih.gov/35523804",
            "pubmed_id": "35523804",
            "year": 2022,
            "authors": "Hoogstraat M, Lips E et al.",
            "author_affiliation": "Netherlands Cancer Institute (NKI‑AVL)",
            "source": "GEO",
            "source_link": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE192341",
            "study_cohorts": "HER2‑negative breast cancer patients receiving neoadjuvant chemotherapy"
        },
        {
            "id": "GSE137867",
            "priority": "Medium",
            "title": (
                "Esophageal cancer tissues before and after radiotherapy (GSE137867)"
            ),
            "summary": (
                "Esophageal squamous cell carcinoma has high mortality in China.  Radiotherapy is a standard treatment, but radiation "
                "resistance contributes to recurrence and metastasis.  This study used microarray‑based gene expression profiling to "
                "identify markers of radioresistance.  Tissue samples from the same patient were collected before and after radiotherapy, "
                "and differential genes were analysed【59533975869721†L33-L44】."),
            "experimental_design": (
                "Four esophageal squamous cell carcinoma patients underwent radiotherapy to 40 Gy.  Tumour tissue samples were collected "
                "immediately before treatment and after completion of radiotherapy.  Paired samples (4 pairs, 8 in total) were profiled "
                "using expression microarrays【59533975869721†L42-L47】."),
            "data_modality": "Expression profiling by array",
            "disease": "Esophageal squamous cell carcinoma",
            "tissue": "Esophageal tumour",
            "sampling_site": "Tumour",
            "treatment_category": "Radiotherapy",
            "treatment_name": "Radiotherapy (40 Gy)",
            "treatment_info": "External beam radiotherapy delivering 40 Gy",
            "treatment_target": "DNA damage via ionising radiation",
            "total_samples": 8,
            "total_patients": 4,
            "sampling_timepoint": "Pre‑ and post‑radiotherapy",
            "patient_distribution": "",
            "matched_pre_post": "Yes",
            "overall_survival_info": "No",
            "clinical_trial_nct_ids": "",
            "trial_phase": "",
            "raw_data_availability": "Yes (GEO)",
            "processed_data_availability": "Yes",
            "publication_availability": "Yes",
            "publication_link": "https://pubmed.ncbi.nlm.nih.gov/33935718",
            "pubmed_id": "33935718",
            "year": 2021,
            "authors": "Yu J, Wang J",
            "author_affiliation": "Affiliated Changzhou No. 2 People’s Hospital, Nanjing Medical University",
            "source": "GEO",
            "source_link": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE137867",
            "study_cohorts": "Four esophageal cancer patients undergoing radiotherapy"
        },
    ]
    df = pd.DataFrame(datasets)
    return df


def filter_datasets(df: pd.DataFrame) -> pd.DataFrame:
    """Apply interactive filters and return the filtered DataFrame."""
    st.sidebar.markdown("## Filter datasets")

    # Text search across title, summary, disease, tissue and treatment name.
    query = st.sidebar.text_input(
        "Search (disease, tissue, treatment name, summary)", "",
        help="Enter keywords to search across multiple fields.")

    # Treatment category filter
    categories = sorted(df['treatment_category'].unique().tolist())
    selected_categories = st.sidebar.multiselect(
        "Treatment category", categories, default=categories)

    # Treatment name filter
    treatment_names = sorted(df['treatment_name'].unique().tolist())
    selected_treatments = st.sidebar.multiselect(
        "Treatment name", treatment_names, default=treatment_names)

    # Disease filter
    diseases = sorted(df['disease'].unique().tolist())
    selected_diseases = st.sidebar.multiselect(
        "Disease", diseases, default=diseases)

    # Tissue filter
    tissues = sorted(df['tissue'].unique().tolist())
    selected_tissues = st.sidebar.multiselect(
        "Tissue", tissues, default=tissues)

    # Treatment target filter (show only unique non‑empty targets)
    targets = sorted([t for t in df['treatment_target'].unique().tolist() if t])
    selected_targets = st.sidebar.multiselect(
        "Treatment target", targets, default=targets)

    # Number of samples filter
    min_samples = int(df['total_samples'].min())
    max_samples = int(df['total_samples'].max())
    sample_range = st.sidebar.slider(
        "Total samples", min_value=min_samples, max_value=max_samples,
        value=(min_samples, max_samples), step=1)

    # Matched pre/post filter
    pre_post_options = ["All", "Yes", "No"]
    selected_pre_post = st.sidebar.radio(
        "Matched pre/post samples", pre_post_options, index=0)

    # Overall survival information filter
    os_options = ["All", "Yes", "No"]
    selected_os = st.sidebar.radio(
        "Overall‑survival information", os_options, index=0)

    # Apply filters
    mask = (
        df['treatment_category'].isin(selected_categories) &
        df['treatment_name'].isin(selected_treatments) &
        df['disease'].isin(selected_diseases) &
        df['tissue'].isin(selected_tissues) &
        df['treatment_target'].isin(selected_targets) &
        (df['total_samples'] >= sample_range[0]) & (df['total_samples'] <= sample_range[1])
    )
    if selected_pre_post != "All":
        mask &= df['matched_pre_post'].eq(selected_pre_post)
    if selected_os != "All":
        mask &= df['overall_survival_info'].eq(selected_os)
    if query:
        # Build a case‑insensitive search across multiple columns.
        q = query.lower()
        mask &= df.apply(
            lambda row: (
                q in str(row['title']).lower()
                or q in str(row['summary']).lower()
                or q in str(row['disease']).lower()
                or q in str(row['tissue']).lower()
                or q in str(row['treatment_name']).lower()
                or q in str(row['treatment_target']).lower()
            ),
            axis=1,
        )
    filtered = df.loc[mask].copy()
    return filtered


def render_table(df: pd.DataFrame) -> None:
    """Render the filtered dataset table with clickable links."""
    if df.empty:
        st.info("No datasets match the selected filters.")
        return

    # Create a column with HTML links to open the detailed view in a new tab.
    def make_link(ds_id: str) -> str:
        params = urllib.parse.urlencode({"dataset_id": ds_id})
        return f'<a href="/?{params}" target="_blank">View</a>'

    df_display = df[
        [
            "id",
            "title",
            "disease",
            "tissue",
            "treatment_category",
            "treatment_name",
            "total_samples",
            "matched_pre_post",
            "overall_survival_info",
        ]
    ].copy()
    df_display.rename(
        columns={
            "id": "Accession",
            "title": "Title",
            "disease": "Disease",
            "tissue": "Tissue",
            "treatment_category": "Treatment category",
            "treatment_name": "Treatment name",
            "total_samples": "Samples",
            "matched_pre_post": "Pre/Post",
            "overall_survival_info": "OS info",
        },
        inplace=True,
    )
    df_display["Details"] = df["id"].apply(make_link)

    # Show metrics above the table
    c1, c2, c3 = st.columns(3)
    c1.metric("Datasets", len(df))
    total_patients_sum = int(df['total_patients'].sum())
    c2.metric("Total patients", total_patients_sum)
    total_samples_sum = int(df['total_samples'].sum())
    c3.metric("Total samples", total_samples_sum)

    # Chart summarising the distribution of treatment categories in the filtered set
    st.subheader("Treatment category distribution")
    cat_counts = df['treatment_category'].value_counts()
    st.bar_chart(cat_counts)

    st.subheader("Datasets")
    # Render DataFrame with HTML for links
    st.write(
        df_display.to_html(escape=False, index=False),
        unsafe_allow_html=True,
    )


def render_dataset_detail(df: pd.DataFrame, dataset_id: str) -> None:
    """Render a detailed view for a single dataset."""
    dataset = df.loc[df['id'] == dataset_id]
    if dataset.empty:
        st.error(f"Dataset '{dataset_id}' not found.")
        return
    ds = dataset.iloc[0].to_dict()

    st.title(ds['title'])
    st.markdown(f"**Accession:** {ds['id']}")
    st.markdown(f"**Treatment category:** {ds['treatment_category']}")
    st.markdown(f"**Treatment name:** {ds['treatment_name']} ({ds['treatment_target']})")
    st.markdown(f"**Disease:** {ds['disease']}")
    st.markdown(f"**Tissue:** {ds['tissue']}")
    st.markdown(f"**Samples:** {ds['total_samples']} | **Patients:** {ds['total_patients']}")
    st.markdown(f"**Sampling timepoint:** {ds['sampling_timepoint']}")
    if ds['patient_distribution']:
        st.markdown(f"**Patient distribution:** {ds['patient_distribution']}")
    st.markdown("\n---\n")

    st.subheader("Summary / Abstract")
    st.write(ds['summary'])

    st.subheader("Experimental design")
    st.write(ds['experimental_design'])

    st.subheader("Treatment information")
    st.write(f"**Treatment category:** {ds['treatment_category']}")
    st.write(f"**Treatment name:** {ds['treatment_name']}")
    st.write(f"**Treatment target:** {ds['treatment_target']}")
    st.write(f"**Treatment details:** {ds['treatment_info']}")

    st.subheader("Data availability and publication")
    st.write(f"**Raw data available:** {ds['raw_data_availability']}")
    st.write(f"**Processed data available:** {ds['processed_data_availability']}")
    st.write(f"**Matched pre/post samples:** {ds['matched_pre_post']}")
    st.write(f"**Overall survival information:** {ds['overall_survival_info']}")
    if ds['publication_availability'] == "Yes":
        if ds['publication_link']:
            st.write(
                f"**Publication:** [PubMed {ds['pubmed_id']}]({ds['publication_link']}) (Year {ds['year']})"
            )
        else:
            st.write(f"**Publication:** PubMed ID {ds['pubmed_id']} (Year {ds['year']})")
    if ds['source_link']:
        st.write(f"**Source:** [GEO Accession page]({ds['source_link']})")

    st.subheader("Authors and affiliations")
    st.write(f"**Authors:** {ds['authors']}")
    st.write(f"**Affiliation:** {ds['author_affiliation']}")

    st.subheader("Study cohorts")
    st.write(ds['study_cohorts'])

    # Back button to return to the main view
    if st.button("Back to dataset list"):
        # Clear query parameters to go back
        st.experimental_set_query_params()


def main() -> None:
    st.set_page_config(
        page_title="Oncology Dataset Explorer",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    df = load_dataset()
    params = st.experimental_get_query_params()
    dataset_id = params.get('dataset_id', [None])[0]
    if dataset_id:
        render_dataset_detail(df, dataset_id)
    else:
        st.title("Oncology Dataset Explorer")
        st.markdown(
            "Use the filters in the sidebar to narrow the list of publicly available oncology datasets. "
            "Each row in the table corresponds to one dataset.  Click **View** to open a detailed description "
            "in a new tab."
        )
        filtered_df = filter_datasets(df)
        render_table(filtered_df)


if __name__ == "__main__":
    main()