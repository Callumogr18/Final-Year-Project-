import logging
import altair as alt
import pandas as pd
import streamlit as st

from DB.db_conn import get_connection

logger = logging.getLogger(__name__)

SCORE_COLS = ['bleu', 'rouge_1', 'rouge_2', 'rouge_l']
SCORE_LABELS = {'bleu': 'BLEU', 'rouge_1': 'ROUGE-1', 'rouge_2': 'ROUGE-2', 'rouge_l': 'ROUGE-L'}
JUDGE_COLS = ['hallucination', 'fluency', 'consistency', 'reasoning', 'coherence', 'accuracy']
JUDGE_LABELS = {
    'hallucination': 'Hallucination',
    'fluency': 'Fluency',
    'consistency': 'Consistency',
    'reasoning': 'Reasoning',
    'coherence': 'Coherence',
    'accuracy': 'Factual Accuracy',
}
PERF_COLS = ['latency', 'tokens_generated', 'tokens_prompt', 'total_tokens']


@st.cache_resource
def get_db_connection():
    conn = get_connection()
    if conn is None:
        st.error("Can't establish connection to DB. Check your .env file and DB status.")
        st.stop()
    return conn



# Note, to avoid hashing of arguments which cache data does, add a leading underscore to _conn param

@st.cache_data
def load_metrics(_conn):
    df = pd.read_sql(
        """
        SELECT m.prompt_id, m.bleu, m.rouge_1, m.rouge_2, m.rouge_l, m.batch_id, m.task_type,
               g.model_name
        FROM metrics m
        JOIN generations g ON m.response_id = g.response_id
        """,
        _conn
    )
    df[SCORE_COLS] = df[SCORE_COLS].apply(pd.to_numeric, errors='coerce')
    return df


@st.cache_data
def load_generation_data(_conn):
    query = """
        SELECT m.prompt_id, m.task_type, m.bleu, m.rouge_1, m.rouge_2, m.rouge_l,
               g.latency, g.tokens_generated, g.tokens_prompt, g.total_tokens, g.model_name
        FROM metrics m
        JOIN generations g ON m.response_id = g.response_id
    """
    df = pd.read_sql(query, _conn)
    df[SCORE_COLS + PERF_COLS] = df[SCORE_COLS + PERF_COLS].apply(pd.to_numeric, errors='coerce')
    return df


@st.cache_data
def load_judge_data(_conn):
    query = """
        SELECT jm.prompt_id, jm.task_type,
               jm.hallucination, jm.fluency, jm.consistency,
               jm.reasoning, jm.coherence, jm.accuracy,
               m.bleu, m.rouge_1, m.rouge_2, m.rouge_l,
               g.llm_response, g.latency, g.tokens_generated, g.tokens_prompt, g.total_tokens,
               g.model_name,
               p.question, p.article
        FROM judge_metrics jm
        JOIN metrics m ON jm.response_id = m.response_id
        JOIN generations g ON jm.response_id = g.response_id
        JOIN prompts p ON jm.prompt_id = p.id
    """
    df = pd.read_sql(query, _conn)
    numeric = JUDGE_COLS + SCORE_COLS + PERF_COLS
    df[numeric] = df[numeric].apply(pd.to_numeric, errors='coerce')
    return df


@st.cache_data
def load_explanations(_conn, prompt_id):
    query = """
        SELECT je.metric, je.question, je.answer, je.explanation, g.model_name
        FROM judge_explanations je
        JOIN generations g ON je.response_id = g.response_id
        WHERE je.prompt_id = %s
        ORDER BY g.model_name, je.metric, je.id
    """
    return pd.read_sql(query, _conn, params=(prompt_id,))


def model_filter(df, key, label="Filter by model"):
    models = sorted(df['model_name'].dropna().unique().tolist())
    selected = st.multiselect(label, options=models, default=models, key=key)
    return df[df['model_name'].isin(selected)] if selected else df


st.set_page_config(page_title="LLM Evaluation Dashboard")#, layout="wide")
st.title("LLM Evaluation Dashboard")

conn = get_db_connection()
metrics_df = load_metrics(conn)

# Lazy-load expensive dataframes using session state
if 'gen_df' not in st.session_state:
    st.session_state.gen_df = None
if 'judge_df' not in st.session_state:
    st.session_state.judge_df = None

tab_overview, tab_metrics, tab_judge, tab_gen, tab_explorer = st.tabs([
    "Overview", "Metric Analysis", "Judge Analysis", "Generation Analysis", "Response Explorer"
])

with tab_overview:
    if st.session_state.judge_df is None:
        with st.spinner("Loading judge data..."):
            st.session_state.judge_df = load_judge_data(conn)
    judge_df = st.session_state.judge_df

    total = len(metrics_df)
    qa_count = len(metrics_df[metrics_df['task_type'].str.upper() == 'QA'])
    summ_count = len(metrics_df[metrics_df['task_type'].str.upper() == 'SUMMARISATION'])
    model_count = metrics_df['model_name'].nunique()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Evaluations", total)
    c2.metric("QA Evaluations", qa_count)
    c3.metric("Summarisation Evaluations", summ_count)
    c4.metric("Models Evaluated", model_count)

    st.divider()

    if not metrics_df.empty:
        st.subheader("Average Scores by Model")
        avg_by_model = (
            metrics_df.groupby('model_name')[SCORE_COLS]
            .mean()
            .reset_index()
            .melt(id_vars='model_name', value_vars=SCORE_COLS, var_name='Metric', value_name='Score')
        )
        avg_by_model['Metric'] = avg_by_model['Metric'].map(SCORE_LABELS)

        chart = alt.Chart(avg_by_model).mark_bar().encode(
            x=alt.X('Metric:N', title=None),
            y=alt.Y('Score:Q', scale=alt.Scale(domain=[0, 1])),
            color=alt.Color('model_name:N', title='Model'),
            xOffset='model_name:N',
            tooltip=['model_name', 'Metric', alt.Tooltip('Score:Q', format='.4f')]
        )
        st.altair_chart(chart, use_container_width=True)

        st.subheader("Average Scores by Task Type")
        avg_by_task = (
            metrics_df.groupby('task_type')[SCORE_COLS]
            .mean()
            .reset_index()
            .melt(id_vars='task_type', value_vars=SCORE_COLS, var_name='Metric', value_name='Score')
        )
        avg_by_task['Metric'] = avg_by_task['Metric'].map(SCORE_LABELS)

        chart = alt.Chart(avg_by_task).mark_bar().encode(
            x=alt.X('Metric:N', title=None),
            y=alt.Y('Score:Q', scale=alt.Scale(domain=[0, 1])),
            color=alt.Color('task_type:N', title='Task Type'),
            xOffset='task_type:N',
            tooltip=['task_type', 'Metric', alt.Tooltip('Score:Q', format='.4f')]
        )
        st.altair_chart(chart, use_container_width=True)

    if not judge_df.empty:
        st.subheader("Average Judge Scores by Model")
        avg_judge_model = (
            judge_df.groupby('model_name')[JUDGE_COLS]
            .mean()
            .reset_index()
            .melt(id_vars='model_name', value_vars=JUDGE_COLS, var_name='Criterion', value_name='Score')
        )
        avg_judge_model['Criterion'] = avg_judge_model['Criterion'].map(JUDGE_LABELS)

        chart = alt.Chart(avg_judge_model).mark_bar().encode(
            x=alt.X('Criterion:N', title=None),
            y=alt.Y('Score:Q', scale=alt.Scale(domain=[0, 1])),
            color=alt.Color('model_name:N', title='Model'),
            xOffset='model_name:N',
            tooltip=['model_name', 'Criterion', alt.Tooltip('Score:Q', format='.4f')]
        )
        st.altair_chart(chart, use_container_width=True)

        st.subheader("Average Judge Scores by Task Type")
        avg_judge = (
            judge_df.groupby('task_type')[JUDGE_COLS]
            .mean()
            .reset_index()
            .melt(id_vars='task_type', value_vars=JUDGE_COLS, var_name='Criterion', value_name='Score')
        )
        avg_judge['Criterion'] = avg_judge['Criterion'].map(JUDGE_LABELS)

        chart = alt.Chart(avg_judge).mark_bar().encode(
            x=alt.X('Criterion:N', title=None),
            y=alt.Y('Score:Q', scale=alt.Scale(domain=[0, 1])),
            color=alt.Color('task_type:N', title='Task Type'),
            xOffset='task_type:N',
            tooltip=['task_type', 'Criterion', alt.Tooltip('Score:Q', format='.4f')]
        )
        st.altair_chart(chart, use_container_width=True)


with tab_metrics:
    if metrics_df.empty:
        st.warning("No metric data found.")
    else:
        filtered_metrics = model_filter(metrics_df, key="metrics_model_filter")

        qa_df = filtered_metrics[filtered_metrics['task_type'].str.upper() == 'QA']
        summ_df = filtered_metrics[filtered_metrics['task_type'].str.upper() == 'SUMMARISATION']

        for label, subset in [("QA", qa_df), ("Summarisation", summ_df)]:
            if subset.empty:
                continue
            st.subheader(f"{label} — Average Scores")
            avg = subset[SCORE_COLS].mean()
            cols = st.columns(4)
            for col, key in zip(cols, SCORE_COLS):
                col.metric(SCORE_LABELS[key], f"{avg[key]:.4f}")

        st.divider()
        st.subheader("Model Comparison by Metric")

        task_choice = st.radio("Task type", ["QA", "Summarisation"], horizontal=True, key="comparison_task")
        comparison_df = qa_df if task_choice == "QA" else summ_df

        if not comparison_df.empty:
            model_avg = (
                comparison_df.groupby('model_name')[SCORE_COLS]
                .mean()
                .reset_index()
                .melt(id_vars='model_name', value_vars=SCORE_COLS, var_name='Metric', value_name='Score')
            )
            model_avg['Metric'] = model_avg['Metric'].map(SCORE_LABELS)

            chart = alt.Chart(model_avg).mark_bar().encode(
                x=alt.X('model_name:N', title=None, axis=alt.Axis(labelAngle=-30)),
                y=alt.Y('Score:Q', scale=alt.Scale(domain=[0, 1])),
                color=alt.Color('model_name:N', title='Model'),
                facet=alt.Facet('Metric:N', columns=2),
                tooltip=['model_name', 'Metric', alt.Tooltip('Score:Q', format='.4f')]
            ).properties(width=280, height=200)
            st.altair_chart(chart)

        _="""
        st.divider()
        st.subheader("Per-Prompt Score Heatmap")

        task_choice2 = st.radio("Task type", ["QA", "Summarisation"], horizontal=True, key="heatmap_task")
        heatmap_df = qa_df if task_choice2 == "QA" else summ_df

        if not heatmap_df.empty:
            heatmap_data = heatmap_df[['prompt_id', 'model_name'] + SCORE_COLS].melt(
                id_vars=['prompt_id', 'model_name'], value_vars=SCORE_COLS, var_name='Metric', value_name='Score'
            )
            heatmap_data['Metric'] = heatmap_data['Metric'].map(SCORE_LABELS)
            heatmap_data['prompt_id'] = heatmap_data['prompt_id'].astype(str)
            heatmap_data['label'] = heatmap_data['prompt_id'] + ' | ' + heatmap_data['model_name']

            heatmap = alt.Chart(heatmap_data).mark_rect().encode(
                x=alt.X('Metric:N', title=None),
                y=alt.Y('label:N', title='Prompt | Model', sort=None),
                color=alt.Color('Score:Q', scale=alt.Scale(domain=[0, 1], scheme='redyellowgreen'), title='Score'),
                tooltip=['prompt_id', 'model_name', 'Metric', alt.Tooltip('Score:Q', format='.4f')]
            ).properties(height=max(200, len(heatmap_data['label'].unique()) * 14))
            st.altair_chart(heatmap, use_container_width=True)
        """
    

with tab_judge:
    if st.session_state.judge_df is None:
        with st.spinner("Loading judge data..."):
            st.session_state.judge_df = load_judge_data(conn)
    judge_df = st.session_state.judge_df

    if judge_df.empty:
        st.warning("No judge data found. Run some evaluations first.")
    else:
        filtered_judge = model_filter(judge_df, key="judge_model_filter")

        qa_j = filtered_judge[filtered_judge['task_type'].str.upper() == 'QA']
        summ_j = filtered_judge[filtered_judge['task_type'].str.upper() == 'SUMMARISATION']

        for label, subset in [("QA", qa_j), ("Summarisation", summ_j)]:
            if subset.empty:
                continue
            st.subheader(f"{label} — Average Judge Scores")
            avg = subset[JUDGE_COLS].mean()
            cols = st.columns(6)
            for col, key in zip(cols, JUDGE_COLS):
                col.metric(JUDGE_LABELS[key], f"{avg[key]:.4f}")

        st.divider()
        st.subheader("Judge Scores by Model")

        task_choice = st.radio("Task type", ["QA", "Summarisation"], horizontal=True, key="judge_model_task")
        judge_model_df = qa_j if task_choice == "QA" else summ_j

        if not judge_model_df.empty:
            model_judge_avg = (
                judge_model_df.groupby('model_name')[JUDGE_COLS]
                .mean()
                .reset_index()
                .melt(id_vars='model_name', value_vars=JUDGE_COLS, var_name='Criterion', value_name='Score')
            )
            model_judge_avg['Criterion'] = model_judge_avg['Criterion'].map(JUDGE_LABELS)

            chart = alt.Chart(model_judge_avg).mark_bar().encode(
                x=alt.X('model_name:N', title=None, axis=alt.Axis(labelAngle=-30)),
                y=alt.Y('Score:Q', scale=alt.Scale(domain=[0, 1])),
                color=alt.Color('model_name:N', title='Model'),
                facet=alt.Facet('Criterion:N', columns=3),
                tooltip=['model_name', 'Criterion', alt.Tooltip('Score:Q', format='.4f')]
            ).properties(width=220, height=180)
            st.altair_chart(chart)

        st.divider()
        st.subheader("Judge vs Traditional Metric Correlation")

        col1, col2 = st.columns(2)
        with col1:
            judge_metric = st.selectbox(
                "Judge criterion",
                options=JUDGE_COLS,
                format_func=lambda k: JUDGE_LABELS[k]
            )
        with col2:
            trad_metric = st.selectbox(
                "Traditional metric",
                options=SCORE_COLS,
                format_func=lambda k: SCORE_LABELS[k]
            )

        scatter = alt.Chart(filtered_judge).mark_circle(size=60, opacity=0.6).encode(
            x=alt.X(f'{trad_metric}:Q', title=SCORE_LABELS[trad_metric], scale=alt.Scale(domain=[0, 1])),
            y=alt.Y(f'{judge_metric}:Q', title=JUDGE_LABELS[judge_metric], scale=alt.Scale(domain=[0, 1])),
            color=alt.Color('model_name:N', title='Model'),
            shape=alt.Shape('task_type:N', title='Task Type'),
            tooltip=[
                'prompt_id', 'task_type', 'model_name',
                alt.Tooltip(f'{trad_metric}:Q', format='.4f'),
                alt.Tooltip(f'{judge_metric}:Q', format='.4f')
            ]
        )
        st.altair_chart(scatter, use_container_width=True)


with tab_gen:
    if st.session_state.gen_df is None:
        with st.spinner("Loading generation data..."):
            st.session_state.gen_df = load_generation_data(conn)
    gen_df = st.session_state.gen_df

    if gen_df.empty:
        st.warning("No generation data found.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            task_filter = st.selectbox("Filter by task type", ["All", "QA", "SUMMARISATION"])
        filtered = gen_df if task_filter == "All" else gen_df[gen_df['task_type'].str.upper() == task_filter]

        filtered = model_filter(filtered, key="gen_model_filter")

        st.subheader("Generation Stats")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Avg Latency (ms)", f"{filtered['latency'].mean():.0f}")
        c2.metric("Avg Tokens Generated", f"{filtered['tokens_generated'].mean():.0f}")
        c3.metric("Avg Prompt Tokens", f"{filtered['tokens_prompt'].mean():.0f}")
        c4.metric("Total Generations", len(filtered))

        st.divider()
        st.subheader("Latency by Model")

        latency_box = alt.Chart(filtered).mark_circle(size=40, opacity=0.6).encode(
            x=alt.X('model_name:N', title='Model', axis=alt.Axis(labelAngle=-30)),
            y=alt.Y('latency:Q', title='Latency (ms)'),
            color=alt.Color('model_name:N', legend=None),
            xOffset='jitter:Q',  # random jitter
            tooltip=['model_name', 'latency']
        ).transform_calculate(
            jitter='sqrt(-2*log(random()))*cos(2*PI*random())'
        )
        st.altair_chart(latency_box, use_container_width=True)

        st.divider()

        melted = filtered.melt(
            id_vars=['task_type', 'model_name', 'latency', 'tokens_generated', 'tokens_prompt'],
            value_vars=SCORE_COLS, var_name='Metric', value_name='Score'
        )
        melted['Metric'] = melted['Metric'].map(SCORE_LABELS)

        st.subheader("Latency vs Score")
        latency_chart = alt.Chart(melted).mark_circle(size=60, opacity=0.7).encode(
            x=alt.X('latency:Q', title='Latency (ms)'),
            y=alt.Y('Score:Q', scale=alt.Scale(domain=[0, 1])),
            color=alt.Color('model_name:N', title='Model'),
            facet=alt.Facet('Metric:N', columns=2),
            tooltip=['model_name', 'task_type', 'latency', 'Score', 'Metric']
        ).properties(width=300, height=200)
        st.altair_chart(latency_chart)

        st.subheader("Tokens Generated vs Score")
        tokens_chart = alt.Chart(melted).mark_circle(size=60, opacity=0.7).encode(
            x=alt.X('tokens_generated:Q', title='Tokens Generated'),
            y=alt.Y('Score:Q', scale=alt.Scale(domain=[0, 1])),
            color=alt.Color('model_name:N', title='Model'),
            facet=alt.Facet('Metric:N', columns=2),
            tooltip=['model_name', 'task_type', 'tokens_generated', 'Score', 'Metric']
        ).properties(width=300, height=200)
        st.altair_chart(tokens_chart)


with tab_explorer:
    if judge_df is None:
        judge_df = load_judge_data(conn)

    st.subheader("Response Explorer")

    search_by = st.radio("Search by", ["Prompt ID", "Batch ID"], horizontal=True)

    if search_by == "Prompt ID":
        prompt_id = st.number_input("Enter Prompt ID", min_value=1, step=1)
        if st.button("Load", key="load_prompt"):
            rows = judge_df[judge_df['prompt_id'] == prompt_id]

            if rows.empty:
                fallback = metrics_df[metrics_df['prompt_id'] == prompt_id]
                if fallback.empty:
                    st.warning(f"No data found for Prompt ID {prompt_id}")
                else:
                    st.dataframe(fallback[['prompt_id', 'model_name', 'task_type'] + SCORE_COLS])
            else:
                r = rows.iloc[0]
                st.markdown(f"**Task Type:** {r['task_type']}")
                if pd.notna(r.get('question')):
                    st.markdown(f"**Question:** {r['question']}")
                if pd.notna(r.get('article')):
                    with st.expander("Article"):
                        st.write(r['article'])

                st.divider()
                st.subheader("Responses by Model")

                explanations_df = load_explanations(conn, int(prompt_id))

                for _, row in rows.iterrows():
                    with st.expander(f"**{row['model_name']}**"):
                        st.markdown("**LLM Response:**")
                        st.info(row['llm_response'])

                        left, right = st.columns(2)
                        with left:
                            st.caption("Traditional Metrics")
                            cols = st.columns(4)
                            for col, key in zip(cols, SCORE_COLS):
                                col.metric(SCORE_LABELS[key], f"{pd.to_numeric(row[key], errors='coerce'):.4f}")
                        with right:
                            st.caption("Judge Scores")
                            cols = st.columns(6)
                            for col, key in zip(cols, JUDGE_COLS):
                                col.metric(JUDGE_LABELS[key], f"{pd.to_numeric(row[key], errors='coerce'):.4f}")

                        model_explanations = explanations_df[explanations_df['model_name'] == row['model_name']]
                        failed = model_explanations[model_explanations['answer'] == False]
                        if not failed.empty:
                            st.divider()
                            st.caption("Failed criteria")
                            for metric, group in failed.groupby('metric'):
                                st.markdown(f"**{metric}**")
                                for _, qa in group.iterrows():
                                    st.markdown(f":x: _{qa['question']}_")
                                    if pd.notna(qa.get('explanation')) and qa['explanation']:
                                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;{qa['explanation']}")

    else:
        batch_id = st.text_input("Enter Batch ID")
        if st.button("Load", key="load_batch") and batch_id:
            batch_data = metrics_df[metrics_df['batch_id'] == batch_id]
            if batch_data.empty:
                st.warning(f"No data found for Batch ID {batch_id}")
            else:
                st.subheader(f"Batch {batch_id} — {len(batch_data)} evaluations")

                model_avg = (
                    batch_data.groupby('model_name')[SCORE_COLS]
                    .mean()
                    .reset_index()
                    .melt(id_vars='model_name', value_vars=SCORE_COLS, var_name='Metric', value_name='Score')
                )
                model_avg['Metric'] = model_avg['Metric'].map(SCORE_LABELS)

                chart = alt.Chart(model_avg).mark_bar().encode(
                    x=alt.X('model_name:N', title=None, axis=alt.Axis(labelAngle=-30)),
                    y=alt.Y('Score:Q', scale=alt.Scale(domain=[0, 1])),
                    color=alt.Color('model_name:N', title='Model'),
                    facet=alt.Facet('Metric:N', columns=2),
                    tooltip=['model_name', 'Metric', alt.Tooltip('Score:Q', format='.4f')]
                ).properties(width=280, height=200)
                st.altair_chart(chart)
