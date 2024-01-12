import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="In Context Data Viz", layout="wide")


@st.cache_resource(max_entries=1)
def load_data():
    print("Loading data...")
    df = pd.read_csv("./numbers_computed.csv")
    return df


df = load_data()

property_to_col = {
    "Modified Answer LogLikelihood": "GPT4 Answer - Modified Context | LogLikelihood",
    "Modified Answer Perplexity": "GPT4 Answer - Modified Context | Perplexity",
    "Original Answer Loglikelihood": "Answer | LogLikelihood",
    "Original Answer Perplexity": "Answer | Perplexity",
    "Token Count of the Modified Answer": "Token Count of the Modified Answer",
    "Token Count of the Modified Context": "Token Count of Modified Context"}

st.sidebar.markdown("## Select a property to visualize")
property_to_show = st.sidebar.radio(
    "Property",
    options=list(property_to_col.keys()),
    index=0)

# Choose which ones among the hues to show
st.sidebar.markdown("## Select a hue to visualize")
st.sidebar.markdown("### Select all that apply")
hues = st.sidebar.multiselect(
    "Hues",
    options=list(df.hue.unique()),
    default=list(df.hue.unique()))


fig, ax = plt.subplots(1, 1, figsize=(7, 4))

xmin = df[property_to_col[property_to_show]].min().astype(float)
xmax = df[property_to_col[property_to_show]].max().astype(float)

x_values = st.slider(
    'Select a range of values for the x-axis:',
    min_value=float(xmin),
    max_value=float(xmax),
    value=(float(xmin), float(xmax)),
    step=0.01)

x_filter = (df[property_to_col[property_to_show]] >= x_values[0]) & (
    df[property_to_col[property_to_show]] <= x_values[1])
x_filter = x_filter & (df.hue.isin(hues))
full_mask = (~(df.hue == "Problem")) & (x_filter)
filtered_df = df[full_mask]
sns.histplot(x=property_to_col[property_to_show],
             hue="hue",
             data=filtered_df,
             stat="density",
             common_norm=False,
             ax=ax)

st.pyplot(fig)
st.markdown("## Some Examples")
samples = filtered_df.sample(5)
for i, row in samples.iterrows():
    st.dataframe(row, use_container_width=True)
