import streamlit as st
import logging
import pandas as pd
import altair as alt

from DB.db_conn import get_connection

logger = logging.getLogger(__name__)

def display_data(df, option):
    if option == 1:
        score_cols = ['bleu', 'rouge_1', 'rouge_2', 'rouge_l']
        melted = df.melt(id_vars='prompt_id', value_vars=score_cols,
                        var_name='Metric', value_name='Score')
        chart = alt.Chart(melted).mark_bar().encode(
            x=alt.X('Score:Q', scale=alt.Scale(domain=[0, 1])),
            y=alt.Y('Metric:N'),
            color=alt.Color('Metric:N', legend=None)
        )
        st.altair_chart(chart, width='stretch')

    if option == 2:
        #st.write(f'{df['task_type']}')
        score_cols = ['bleu', 'rouge_1', 'rouge_2', 'rouge_l']
        melted = df.melt(id_vars='prompt_id', value_vars=score_cols,
                        var_name='Metric', value_name='Score')
        chart = alt.Chart(melted).mark_bar().encode(
            x=alt.X('Score:Q', scale=alt.Scale(domain=[0, 1])),
            y=alt.Y('Metric:N'),
            color=alt.Color('Metric:N', legend=None),
            row=alt.Row('prompt_id:N', title='Prompt ID')
        )
        st.altair_chart(chart)


def pull_data(conn, prompt_id=None, batch_id=None):
    if prompt_id is not None:
        query = "SELECT prompt_id, bleu, rouge_1, rouge_2, rouge_l, batch_id FROM metrics WHERE prompt_id = %s;"
        return pd.read_sql(query, conn, params=(prompt_id,))
    elif batch_id is not None:
        query = "SELECT prompt_id, bleu, rouge_1, rouge_2, rouge_l, batch_id FROM metrics WHERE batch_id = %s;"
        return pd.read_sql(query, conn, params=(batch_id,))


def establish_connection():
    conn = get_connection()

    if conn is None:
        st.error("Can't establish connection to DB. Check your .env file and DB status.")
        st.stop()

    return conn

st.title("Data Visualisations")

display_choice = st.radio(
    label="Select data to display",
    options=["By Prompt ID", "By Batch ID"],
    captions=[
        "Returns evaluations for that Prompt ID", 
        "Returns evaluations for that Batch ID"
    ]    
)

if display_choice == "By Prompt ID":
    prompt_id = st.number_input("Enter Prompt ID", step=1)
    conn = establish_connection()
    data = pull_data(conn, prompt_id=prompt_id)
    if data.empty:
        st.warning(f"No data found for {prompt_id}")
    display_data(data, 1)

if display_choice == "By Batch ID":
    batch_id = st.text_input("Enter Batch ID")
    conn = establish_connection()
    data = pull_data(conn, batch_id=batch_id)
    if data.empty:
        st.warning(f"No data found for {batch_id}")
    display_data(data, 2)