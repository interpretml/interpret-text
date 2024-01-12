import streamlit as st
import numpy as np

css_highlight = """
<style>
div.highlight {
    background-color: #fcefe8;
    padding: 10px;
    border-radius: 5px;
}
</style>
"""

css = """
<style>
div.highlight {
    background-color: #fcefe8;
    padding: 10px;
    border-radius: 5px;
}
</style>
"""


def start_frontend():
    st.set_page_config(
        page_title="Hierarchically Explaining LLM completions",
        layout="wide")
    st.markdown(css_highlight, unsafe_allow_html=True)


def get_color(importance_value):
    """
    Converts importance value to a color gradient: red for negative, green for positive.
    """
    if importance_value < 0:
        color = f'#ff{int(255 + 255 * importance_value):02x}{int(255 + 255 * importance_value):02x}'
    else:
        color = f'#{int(255 - 255 * importance_value):02x}ff{int(255 - 255 * importance_value):02x}'
    return color


def explanation_text(parts, attribution_scores):
    importance = np.zeros(len(parts))
    for (feat_idx, coef) in attribution_scores:
        importance[feat_idx] = coef

    importance_normalized = importance / (np.max(np.abs(importance)) + 1e-10)
    explanation_text = []

    for feat_idx in range(len(parts)):
        color = get_color(importance_normalized[feat_idx])
        coef = importance[feat_idx]
        ngram_text = parts[feat_idx]
        ngram_text = ngram_text.replace("\n", "<br>")
        if np.isclose(coef, 0):
            explanation_text.append(
                f'<span style="color:#000000;font-size:16px;border-radius:2%;">{ngram_text}</span>')

        else:
            explanation_text.append(
                f'<span style="background-color:{color};color:#000000;'
                f'font-size:16px;border-radius:2%;">{ngram_text} '
                f'({coef:.2f})</span>'
            )

    explanation_text = " ".join(explanation_text)
    return explanation_text


class ProgressBar:
    def __init__(self, iterable):
        self.bar = st.progress(0, text="Running perturbations")
        self.iterable = iterable
        self.total = len(iterable)
        self.progress = 0
        self._iter = iter(self.iterable)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            val = next(self._iter)
            self.progress += 1
            self.bar.progress(
                self.progress / self.total,
                text=f"Running perturbations ({self.progress}/{self.total})")
            return val
        except StopIteration:
            self.bar.empty()
            raise
