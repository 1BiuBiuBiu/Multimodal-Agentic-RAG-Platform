"""Save variant configurations, launch one batch and inspect comparable scores."""
import json
import httpx
import pandas as pd
import streamlit as st

from app.config_ui import config_controls, LABELS
from app.pipeline_config import PipelineConfig, default_variants


def _request(api_base, method, path, **kwargs):
    response = httpx.request(method, api_base + path, trust_env=False, timeout=20, **kwargs)
    try:
        data = response.json()
    except ValueError:
        raise ValueError(f"接口 {path} 返回空内容或非 JSON 响应（HTTP {response.status_code}），请检查后端服务。")
    if not response.is_success:
        detail = data.get("detail", response.text)
        raise ValueError(str(detail))
    return data


def _parse_dataset(text):
    if not text.strip():
        return [], "评估题集为空，请填写题集、导入 JSON，或点击“恢复项目题集”。已有评估任务会继续运行。"
    try:
        dataset = json.loads(text)
    except json.JSONDecodeError as exc:
        return [], f"题集 JSON 格式错误：第 {exc.lineno} 行、第 {exc.colno} 列。请检查括号、引号和逗号。"
    if not isinstance(dataset, list) or not dataset:
        return [], "题集必须是非空 JSON 数组。"
    for index, row in enumerate(dataset, 1):
        if not isinstance(row, dict) or any(not isinstance(row.get(key), str) or not row[key].strip()
                                            for key in ("question", "ground_truth")):
            return [], f"第 {index} 道题必须包含非空的 question 和 ground_truth 字符串。"
    return dataset, None


def _sync_dataset_editor():
    # Durable state must not share the key of a widget: Streamlit removes widget
    # keys when the user switches to a view where that widget is not rendered.
    st.session_state.ablation_dataset_text = st.session_state._ablation_dataset_editor


def _set_dataset(text):
    st.session_state.ablation_dataset_text = text
    st.session_state._ablation_dataset_editor = text


def _save(api_base, variants):
    saved = _request(api_base, "PUT", "/evaluations/variants", json=variants)["variants"]
    st.session_state.ablation_variants = saved


def _job_panel(api_base, job_id):
    try:
        job = _request(api_base, "GET", f"/evaluations/jobs/{job_id}")
    except (httpx.HTTPError, ValueError) as exc:
        st.error(f"无法读取进度：{exc}")
        return
    active = job["status"] in ("queued", "running")
    st.subheader("批次进度与结果")
    st.write(f"状态：{job['status']}")
    if job.get("error"):
        st.error(job["error"])
    total = len(job["request"]["dataset"]) * len(job["results"])
    completed = sum(row["completed_questions"] for row in job["results"].values())
    st.progress(completed / total, text=f"已处理问题 {completed} / {total}；完成检索后还需等待指标评分。")
    table = []
    for name, result in job["results"].items():
        metrics = result.get("metrics", {})
        row_dict = {
            "变体": name,
            "状态": result["status"],
            "阶段": result["phase"],
            "完成题数": result["completed_questions"],
            "平均检索次数": metrics.get("avg_retrieval_count"),
            "平均耗费Token": metrics.get("avg_tokens"),
            "平均耗时(ms)": metrics.get("latency_ms"),
        }
        for k, v in metrics.items():
            if k not in ("avg_retrieval_count", "latency_ms", "avg_tokens"):
                row_dict[k] = v
        table.append(row_dict)
        if result.get("error"):
            st.error(f"{name}：{result['error']}")
    st.dataframe(pd.DataFrame(table), hide_index=True, use_container_width=True)
    if any(value is None for result in job["results"].values() for value in result.get("metrics", {}).values()):
        st.warning("部分指标未取得完整评分，已留空；请查看有效评分数量或重新评估。")
    if job.get("snapshot_id"):
        st.caption(f"索引快照：{job['snapshot_id'][:16]} · 题集：{job['dataset_hash'][:16]} · top_k 在本批保持一致。")
    st.caption("RAGAs context precision/recall 是基于参考答案的模型评分；精确块级指标仅统计有相关块标注的问题。缺失评分显示为空，不按 0 填充。并发耗时用于本批参考，不是独立延迟基准。")
    if not active:
        names = list(job["results"])
        baseline = st.selectbox("差值对比基线", names, key=f"baseline_{job_id}")
        base = job["results"][baseline].get("metrics", {})
        deltas = []
        for name, row in job["results"].items():
            values = {f"Δ {metric}": round(value-base[metric], 5)
                      for metric, value in row.get("metrics", {}).items()
                      if metric != "latency_ms" and isinstance(value, (int, float)) and isinstance(base.get(metric), (int, float))
                      and row.get("valid_counts", {}).get(metric) == job["results"][baseline].get("valid_counts", {}).get(metric)}
            deltas.append({"变体": name, **values})
        st.dataframe(pd.DataFrame(deltas), hide_index=True, use_container_width=True)
    with st.expander("逐题检索轨迹、回答和有效评分数量"):
        chosen = st.selectbox("查看变体", list(job["results"]), key=f"inspect_{job_id}")
        result = job["results"][chosen]
        m = result.get("metrics", {})
        if m:
            cols1 = st.columns(3)
            cols1[0].metric("平均检索次数", f"{m.get('avg_retrieval_count', '-')}")
            cols1[1].metric("平均消耗 Token", f"{m.get('avg_tokens', '-')}")
            cols1[2].metric("平均延迟", f"{m.get('latency_ms', '-')} ms")

            cols2 = st.columns(4)
            cols2[0].metric("真实性 (Faithfulness)", f"{m.get('faithfulness', '-')}")
            cols2[1].metric("回答相关性 (Answer Relevance)", f"{m.get('response_relevancy', '-')}")
            cols2[2].metric("上下文精确率 (Precision)", f"{m.get('llm_context_precision_with_reference', m.get('context_precision', '-'))}")
            cols2[3].metric("上下文召回率 (Recall)", f"{m.get('llm_context_recall', m.get('context_recall', '-'))}")
        st.write("各指标有效样本数", result.get("valid_counts", {}))
        st.json(result.get("samples", []), expanded=False)
    st.download_button("下载完整结果 JSON", json.dumps(job, ensure_ascii=False, indent=2),
                       f"evaluation-{job_id}.json", mime="application/json")
    st.download_button("下载指标 CSV", pd.DataFrame(table).to_csv(index=False).encode("utf-8-sig"),
                       f"metrics-{job_id}.csv", mime="text/csv")
    if st.session_state.get("ablation_active") and not active:
        st.session_state.ablation_active = False
        st.rerun()


def render_experiments(api_base):
    st.header("🧪 检索模块消融评估")
    st.caption("先逐一添加变体，再用同一组问题、同一索引快照与相同 top_k 并行比较。")
    try:
        if "ablation_variants" not in st.session_state:
            st.session_state.ablation_variants = _request(api_base, "GET", "/evaluations/variants")["variants"]
        if "ablation_dataset_text" not in st.session_state:
            legacy = st.session_state.get("ablation_dataset", "")
            if isinstance(legacy, str) and legacy.strip():
                st.session_state.ablation_dataset_text = legacy
            else:
                dataset = _request(api_base, "GET", "/evaluations/dataset")["dataset"]
                st.session_state.ablation_dataset_text = json.dumps(dataset, ensure_ascii=False, indent=2)
        if "_ablation_dataset_editor" not in st.session_state:
            st.session_state._ablation_dataset_editor = st.session_state.ablation_dataset_text
        with st.expander("添加或更新一个变体"):
            name = st.text_input("变体名称", value="自定义变体")
            values = config_controls("variant")
            if st.button("添加 / 更新变体"):
                config = PipelineConfig.model_validate(values)
                variants = [v for v in st.session_state.ablation_variants if v["name"] != name.strip()]
                _save(api_base, variants + [{"name": name.strip(), "config": config.model_dump()}])
                st.success("配置已保存。")
        if st.button("载入五组消融预设"):
            _save(api_base, default_variants())
        variants = [{"name": v["name"], "config": PipelineConfig.model_validate(v["config"]).model_dump()}
                    for v in st.session_state.ablation_variants]
        st.session_state.ablation_variants = variants
        st.dataframe(pd.DataFrame([{"变体": v["name"], **{LABELS.get(k, k): value for k, value in v["config"].items()}}
                                   for v in variants]), hide_index=True, use_container_width=True)
        remove = st.selectbox("删除变体", [v["name"] for v in variants])
        if st.button("删除所选变体", disabled=len(variants) <= 1):
            _save(api_base, [v for v in variants if v["name"] != remove])
            st.rerun()
        with st.expander("评估题集与相关块标注"):
            uploaded = st.file_uploader("导入题集 JSON", type="json", key="eval_dataset_upload")
            if uploaded and st.button("载入上传题集"):
                try:
                    text = uploaded.getvalue().decode("utf-8-sig")
                    parsed, error = _parse_dataset(text)
                    if error:
                        st.error(f"导入失败：{error}")
                    else:
                        _set_dataset(json.dumps(parsed, ensure_ascii=False, indent=2))
                except UnicodeDecodeError:
                    st.error("题集文件需要使用 UTF-8 编码。")
            if st.button("恢复项目题集"):
                dataset = _request(api_base, "GET", "/evaluations/dataset")["dataset"]
                _set_dataset(json.dumps(dataset, ensure_ascii=False, indent=2))
            st.text_area("题集 JSON", height=300, key="_ablation_dataset_editor", on_change=_sync_dataset_editor)
            st.caption('每题包含 question、ground_truth，可选 relevant_chunk_ids，例如 ["original.pdf::1"]。块 ID 可在 PDF 分块界面查看。未标注时不会计算块级 Recall@k 等指标，也不会将参考答案当作块标签。')
        dataset, dataset_error = _parse_dataset(st.session_state.ablation_dataset_text)
        if dataset_error:
            st.error(dataset_error)
        selected = st.multiselect("本批评估的变体", [v["name"] for v in variants], default=[v["name"] for v in variants])
        count = st.number_input("本次使用前 N 个问题", 1, max(1, len(dataset)), min(3, max(1, len(dataset))), disabled=bool(dataset_error))
        scope = st.radio("评估范围", ["仅检索", "完整 RAG"], horizontal=True)
        st.caption("仅检索：不运行路由、相关性判断、改写及网页回退，比较初次检索结果。完整 RAG：按各变体开关运行完整流程，并额外评分回答忠实度和相关性。")
        use_ragas = st.checkbox("启用 RAGAs 模型评分（会调用 DeepSeek）", value=True)
        concurrency = st.slider("并行变体数", 1, 5, 2)
        top_k = st.slider("本批统一 top_k", 1, 20, 5)
        candidate_k = st.slider("本批统一候选块数", 20, 100, 20, step=5)
        st.info(f"将评估 {len(selected)} 个变体 × {count} 个问题；并行度 {concurrency}。运行记录及完整配置将保存在本机。")
        if st.button("开始批量并行评估", type="primary", disabled=bool(dataset_error) or not selected or st.session_state.get("ablation_active", False)):
            chosen = [{"name": v["name"], "config": {**v["config"], "top_k": top_k, "candidate_k": candidate_k}}
                      for v in variants if v["name"] in selected]
            job = _request(api_base, "POST", "/evaluations/jobs", json={
                "variants": chosen, "dataset": dataset[:count], "scope": "retrieval" if scope == "仅检索" else "full",
                "concurrency": concurrency, "use_ragas": use_ragas,
            })
            st.session_state.ablation_job = job["id"]
            st.session_state.ablation_active = True
        with st.expander("查看历史批次"):
            if st.button("刷新历史批次"):
                st.session_state.ablation_history = _request(api_base, "GET", "/evaluations/jobs")["jobs"]
            history = st.session_state.get("ablation_history", [])
            if history:
                old_id = st.selectbox("历史批次", [j["id"] for j in history])
                if st.button("打开批次"):
                    st.session_state.ablation_job = old_id
                    st.session_state.ablation_active = True
    except (httpx.HTTPError, ValueError) as exc:
        st.error(f"评估配置或请求失败：{exc}")
    # A draft configuration error must never hide an already-running batch.
    if st.session_state.get("ablation_job"):
        panel = st.fragment(run_every=3 if st.session_state.get("ablation_active") else None)(_job_panel)
        panel(api_base, st.session_state.ablation_job)
