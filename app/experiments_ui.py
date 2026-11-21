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
    normalized = []
    for index, row in enumerate(dataset, 1):
        if not isinstance(row, dict) or any(not isinstance(row.get(key), str) or not row[key].strip()
                                            for key in ("question", "ground_truth")):
            return [], f"第 {index} 道题必须包含非空的 question 和 ground_truth 字符串。"
        q_type = row.get("question_type", "simple")
        if q_type not in ("simple", "complex"):
            return [], f"第 {index} 道题的 question_type 只能为 'simple' 或 'complex'。"
        normalized.append({**row, "question_type": q_type})
    return normalized, None


def _sync_dataset_editor():
    # Durable state must not share the key of a widget: Streamlit removes widget
    # keys when the user switches to a view where that widget is not rendered.
    st.session_state.ablation_dataset_text = st.session_state._ablation_dataset_editor
    st.session_state._ablation_user_edited = True


def _set_dataset(text, *, user_edited: bool = False):
    st.session_state.ablation_dataset_text = text
    st.session_state._ablation_dataset_editor = text
    st.session_state._ablation_user_edited = user_edited


def _save(api_base, variants):
    saved = _request(api_base, "PUT", "/evaluations/variants", json=variants)["variants"]
    st.session_state.ablation_variants = saved


def _metrics_for_category(result: dict, category: str) -> dict:
    by_type = result.get("metrics_by_type") or {}
    if category in by_type and by_type[category]:
        return by_type[category]
    if category == "overall":
        return result.get("metrics", {})
    return {}


def _valid_counts_for_category(result: dict, category: str) -> dict:
    by_type = result.get("valid_counts_by_type") or {}
    if category in by_type and by_type[category]:
        return by_type[category]
    if category == "overall":
        return result.get("valid_counts", {})
    return {}


def _format_metric_row(label: str, result: dict, m: dict, extra: dict = None) -> dict:
    row_dict = {"变体": label}
    if extra:
        row_dict.update(extra)
    row_dict.update({
        "状态": result["status"],
        "阶段": result["phase"],
        "题数": m.get("question_count", result["completed_questions"]),
        "平均检索轮次": m.get("avg_retrieval_count"),
        "平均耗费Token": m.get("avg_tokens"),
        "平均耗时(ms)": m.get("latency_ms"),
    })
    for k, v in m.items():
        if k not in ("avg_retrieval_count", "latency_ms", "avg_tokens", "question_count"):
            row_dict[k] = v
    return row_dict


def _fmt_metric_card_val(val) -> str:
    return "-" if val is None else f"{val}"


def _render_metric_cards(title: str, m: dict):
    if not m:
        return
    st.markdown(f"**{title}**（共 {m.get('question_count', '-')} 题）")
    cols1 = st.columns(3)
    cols1[0].metric("平均检索轮次", _fmt_metric_card_val(m.get('avg_retrieval_count')))
    cols1[1].metric("平均消耗 Token", _fmt_metric_card_val(m.get('avg_tokens')))
    latency = m.get('latency_ms')
    cols1[2].metric("平均延迟", "-" if latency is None else f"{latency} ms")

    cols2 = st.columns(5)
    cols2[0].metric("答案完备度 (Correctness)", _fmt_metric_card_val(m.get('answer_correctness')))
    relevancy = m.get('answer_relevancy') if 'answer_relevancy' in m else m.get('response_relevancy')
    cols2[1].metric("回答相关性 (Relevancy)", _fmt_metric_card_val(relevancy))
    cols2[2].metric("真实性 (Faithfulness)", _fmt_metric_card_val(m.get('faithfulness')))
    prec = m.get('llm_context_precision_with_reference') if 'llm_context_precision_with_reference' in m else m.get('context_precision')
    cols2[3].metric("上下文精确率 (Precision)", _fmt_metric_card_val(prec))
    rec = m.get('llm_context_recall') if 'llm_context_recall' in m else m.get('context_recall')
    cols2[4].metric("上下文召回率 (Recall)", _fmt_metric_card_val(rec))


def _job_panel(api_base, job_id):
    try:
        job = _request(api_base, "GET", f"/evaluations/jobs/{job_id}")
    except (httpx.HTTPError, ValueError) as exc:
        st.error(f"无法读取进度：{exc}")
        return
    active = job["status"] in ("queued", "running")
    st.subheader("批次进度与结果")
    st.write(f"状态：{job['status']}")
    if active:
        if st.button("⏹️ 停止当前评估", key=f"stop_panel_btn_{job_id}", type="primary"):
            _request(api_base, "POST", f"/evaluations/jobs/{job_id}/cancel")
            st.session_state.ablation_active = False
            st.rerun()
    shown_errors: list[str] = []
    job_err = (job.get("error") or "").strip()
    if job_err:
        shown_errors.append(job_err)
    for v_name, res in job["results"].items():
        v_err = (res.get("error") or "").strip()
        if not v_err or (job_err and v_err == job_err):
            continue
        formatted = f"【{v_name}】{v_err}"
        if formatted not in shown_errors:
            shown_errors.append(formatted)
    for err_msg in shown_errors:
        st.error(err_msg)
    has_any_samples = any(bool(res.get("samples")) for res in job["results"].values())
    is_ragas_scored = any(
        res.get("ragas_scored") is True or any(k in (res.get("metrics") or {}) for k in (
            "faithfulness", "answer_relevancy", "response_relevancy",
            "answer_correctness", "context_recall", "llm_context_recall",
            "context_precision", "llm_context_precision_with_reference",
        ))
        for res in job["results"].values()
    )
    if not active and has_any_samples:
        ragas_conc_now = min(max(1, int(st.session_state.get("ablation_ragas_concurrency", 16) or 16)), 64)
        if not is_ragas_scored:
            st.info("💡 本批次已完成检索与回答生成，尚未执行 RAGAs 模型评分。您可以随时点击下方按钮发起高并发 RAGAs 评分。")
            if st.button(f"⚡ 立即对本批已生成结果执行高并发 RAGAs 评分（并发度 {ragas_conc_now}）", key=f"ragas_score_btn_{job_id}", type="primary"):
                try:
                    _request(api_base, "POST", "/evaluations/jobs/ragas", json={
                        "job_ids": [job_id],
                        "ragas_concurrency": ragas_conc_now,
                    })
                    st.session_state.ablation_active = True
                    st.rerun()
                except (httpx.HTTPError, ValueError) as exc:
                    st.error(f"启动 RAGAs 评分失败：{exc}")
        else:
            metric_options = [
                "context_recall",
                "llm_context_precision_with_reference",
                "answer_correctness",
                "faithfulness",
                "answer_relevancy",
            ]
            metric_labels = {
                "context_recall": "context_recall（上下文召回率 · 语义原子要点）",
                "llm_context_precision_with_reference": "llm_context_precision_with_reference（子问题分组精确率）",
                "answer_correctness": "answer_correctness（答案完备度与准确度）",
                "faithfulness": "faithfulness（忠实度）",
                "answer_relevancy": "answer_relevancy（回答相关性）",
            }
            chosen_metrics = st.multiselect(
                "选择需要重算的 RAGAs 指标（未选指标保持已有评分不变）",
                metric_options,
                default=["context_recall", "llm_context_precision_with_reference", "answer_correctness"],
                format_func=lambda k: metric_labels.get(k, k),
                key=f"ragas_selected_metrics_{job_id}",
            )
            if st.button(
                f"🚀 重算所选 RAGAs 指标（并发度 {ragas_conc_now}）",
                key=f"ragas_score_btn_{job_id}",
                type="primary",
                disabled=not chosen_metrics,
            ):
                try:
                    _request(api_base, "POST", "/evaluations/jobs/ragas", json={
                        "job_ids": [job_id],
                        "ragas_concurrency": ragas_conc_now,
                        "selected_metrics": chosen_metrics,
                    })
                    st.session_state.ablation_active = True
                    st.rerun()
                except (httpx.HTTPError, ValueError) as exc:
                    st.error(f"启动 RAGAs 评分失败：{exc}")
    total = len(job["request"]["dataset"]) * len(job["results"])
    completed = sum(row["completed_questions"] for row in job["results"].values())
    st.progress(completed / total if total else 0.0, text=f"已处理问题 {completed} / {total}；完成检索后还需等待指标评分。")

    ragas_total = sum(int(row.get("ragas_total_steps") or 0) for row in job["results"].values())
    ragas_done = sum(int(row.get("ragas_completed_steps") or 0) for row in job["results"].values())
    if ragas_total > 0:
        r_ratio = min(1.0, max(0.0, ragas_done / ragas_total))
        r_pct = int(r_ratio * 100)
        st.progress(
            r_ratio,
            text=f"📊 RAGAs 模型评分进度：已完成 {ragas_done} / {ragas_total} 项指标评估 ({r_pct}%)",
        )

    view_choice = st.radio(
        "指标统计维度",
        ["📊 总体汇总 (Overall)", "🟢 简单问题 (Simple)", "🟣 复杂问题 (Complex)", "⚖️ 简单 vs 复杂同表对比"],
        horizontal=True,
        key=f"metric_view_{job_id}",
    )
    cat_map = {
        "📊 总体汇总 (Overall)": "overall",
        "🟢 简单问题 (Simple)": "simple",
        "🟣 复杂问题 (Complex)": "complex",
        "⚖️ 简单 vs 复杂同表对比": "compare",
    }
    category = cat_map[view_choice]

    table = []
    export_rows = []
    for name, result in job["results"].items():
        if category == "compare":
            for sub_cat, tag in (("simple", "🟢 简单"), ("complex", "🟣 复杂")):
                m = _metrics_for_category(result, sub_cat)
                if m:
                    table.append(_format_metric_row(name, result, m, extra={"题型": tag}))
            if not any(_metrics_for_category(result, sc) for sc in ("simple", "complex")):
                table.append(_format_metric_row(name, result, _metrics_for_category(result, "overall"), extra={"题型": "全部"}))
        else:
            m = _metrics_for_category(result, category)
            table.append(_format_metric_row(name, result, m))

        for sub_cat, tag in (("overall", "总体"), ("simple", "简单"), ("complex", "复杂")):
            m = _metrics_for_category(result, sub_cat)
            if m or sub_cat == "overall":
                export_rows.append(_format_metric_row(name, result, m, extra={"题型维度": tag}))

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
        delta_cat = category if category in ("overall", "simple", "complex") else "overall"
        base = _metrics_for_category(job["results"][baseline], delta_cat)
        base_valid = _valid_counts_for_category(job["results"][baseline], delta_cat)
        deltas = []
        for name, row in job["results"].items():
            row_m = _metrics_for_category(row, delta_cat)
            row_valid = _valid_counts_for_category(row, delta_cat)
            values = {f"Δ {metric}": round(value - base[metric], 5)
                      for metric, value in row_m.items()
                      if metric not in ("latency_ms", "question_count")
                      and isinstance(value, (int, float)) and isinstance(base.get(metric), (int, float))
                      and row_valid.get(metric) == base_valid.get(metric)}
            deltas.append({"变体": name, **values})
        st.dataframe(pd.DataFrame(deltas), hide_index=True, use_container_width=True)
    with st.expander("逐题检索轨迹、回答和有效评分数量"):
        chosen = st.selectbox("查看变体", list(job["results"]), key=f"inspect_{job_id}")
        result = job["results"][chosen]
        m_overall = _metrics_for_category(result, "overall")
        m_simple = _metrics_for_category(result, "simple")
        m_complex = _metrics_for_category(result, "complex")
        _render_metric_cards("📊 总体汇总 (Overall)", m_overall)
        if m_simple:
            st.divider()
            _render_metric_cards("🟢 简单问题 (Simple)", m_simple)
        if m_complex:
            st.divider()
            _render_metric_cards("🟣 复杂问题 (Complex)", m_complex)
        st.write("各指标有效样本数", result.get("valid_counts_by_type") or result.get("valid_counts", {}))
        sample_filter = st.radio("样本类型筛选", ["全部", "简单问题", "复杂问题"], horizontal=True, key=f"sample_filter_{job_id}")
        samples = result.get("samples", [])
        if sample_filter == "简单问题":
            samples = [s for s in samples if s.get("question_type", "simple") == "simple"]
        elif sample_filter == "复杂问题":
            samples = [s for s in samples if s.get("question_type") == "complex"]
        if not samples:
            st.info("暂无符合条件的逐题样本数据。")
        else:
            for idx, sample in enumerate(samples, 1):
                q_badge = "🟣 复杂问题" if sample.get("question_type") == "complex" else "🟢 简单问题"
                is_llm_only = sample.get("route") == "llm_only" or (
                    sample.get("retrieval_count", 1) == 0
                    and not sample.get("sources")
                    and not sample.get("web_contexts")
                )
                route_badge = " · 🧠 llm_only 免检索直接回答" if is_llm_only else ""
                with st.container(border=True):
                    st.markdown(f"**[第 {idx} 题 · {q_badge}{route_badge}]** {sample.get('question', '')}")
                    m_scores = sample.get("metrics") or {}
                    rounds_str = "0 (免检索)" if is_llm_only else f"{sample.get('retrieval_count', 1)}"
                    stat_parts = [
                        f"检索轮次: {rounds_str}",
                        f"Token: {sample.get('total_tokens', 0)}",
                        f"耗时: {sample.get('latency_ms', 0):.1f} ms",
                    ]
                    for m_k, m_v in m_scores.items():
                        stat_parts.append(f"{m_k}: {m_v:.3f}" if isinstance(m_v, (int, float)) else f"{m_k}: -")
                    st.caption(" | ".join(stat_parts))
                    col_ans, col_gt = st.columns(2)
                    with col_ans:
                        st.markdown("**🤖 模型生成回答**")
                        st.write(sample.get("answer") or "（仅检索模式，未生成回答）")
                    with col_gt:
                        st.markdown("**🎯 标准参考答案**")
                        st.write(sample.get("ground_truth", ""))
                    queries = [q for t in (sample.get("trace") or []) for q in (t.get("queries") or [])]
                    chunk_ids = sample.get("retrieved_chunk_ids") or []
                    trace_info = []
                    if queries:
                        trace_info.append("检索查询: " + " / ".join(queries[:5]))
                    if chunk_ids:
                        trace_info.append("命中切片: " + ", ".join(chunk_ids[:8]))
                    if trace_info:
                        st.caption(" · ".join(trace_info))
    st.download_button("下载完整结果 JSON", json.dumps(job, ensure_ascii=False, indent=2),
                       f"evaluation-{job_id}.json", mime="application/json")
    st.download_button("下载指标 CSV", pd.DataFrame(export_rows or table).to_csv(index=False).encode("utf-8-sig"),
                       f"metrics-{job_id}.csv", mime="text/csv")
    if st.session_state.get("ablation_active") and not active:
        st.session_state.ablation_active = False
        st.rerun()


def _render_dataset_preview(dataset: list[dict]):
    """Render categorized Simple / Complex question preview table and card detail viewer."""
    if not dataset:
        return
    st.subheader("📋 测评题库分类预览")
    simple_count = sum(1 for q in dataset if q.get("question_type", "simple") == "simple")
    complex_count = sum(1 for q in dataset if q.get("question_type") == "complex")

    m_cols = st.columns(3)
    m_cols[0].metric("📚 题库总题数", f"{len(dataset)} 题")
    m_cols[1].metric("🟢 简单问题 (Simple)", f"{simple_count} 题")
    m_cols[2].metric("🟣 复杂问题 (Complex)", f"{complex_count} 题")

    filter_choice = st.radio(
        "题目分类查看",
        ["📚 全部题目", "🟢 简单问题 (Simple)", "🟣 复杂问题 (Complex)"],
        horizontal=True,
        key="ablation_preview_filter",
    )
    indexed_rows = []
    for idx, q in enumerate(dataset, 1):
        q_type = q.get("question_type", "simple")
        if filter_choice == "🟢 简单问题 (Simple)" and q_type != "simple":
            continue
        if filter_choice == "🟣 复杂问题 (Complex)" and q_type != "complex":
            continue
        indexed_rows.append((idx, q_type, q))

    if not indexed_rows:
        st.info("当前筛选分类下暂无题目。")
        return

    preview_table = [
        {
            "题号": idx,
            "题型": "🟣 复杂问题" if q_type == "complex" else "🟢 简单问题",
            "问题 (Question)": q["question"],
            "标准答案 (Ground Truth)": q["ground_truth"],
        }
        for idx, q_type, q in indexed_rows
    ]
    st.dataframe(pd.DataFrame(preview_table), hide_index=True, use_container_width=True, height=280)

    label_map = {
        idx: f"第 {idx} 题 · {'🟣 复杂' if q_type == 'complex' else '🟢 简单'} · {q['question'][:48]}{'…' if len(q['question']) > 48 else ''}"
        for idx, q_type, q in indexed_rows
    }
    chosen_idx = st.selectbox(
        "选择题目查看完整问答详情",
        [idx for idx, _, _ in indexed_rows],
        format_func=lambda i: label_map[i],
        key="ablation_preview_item",
    )
    _, chosen_type, chosen_q = next(row for row in indexed_rows if row[0] == chosen_idx)
    with st.container(border=True):
        type_tag = "🟣 复杂问题 (Complex)" if chosen_type == "complex" else "🟢 简单问题 (Simple)"
        st.caption(f"第 {chosen_idx} 题 · {type_tag}")
        c_left, c_right = st.columns(2)
        with c_left:
            st.markdown("**❓ 问题全文**")
            st.write(chosen_q["question"])
        with c_right:
            st.markdown("**🎯 标准参考答案**")
            st.write(chosen_q["ground_truth"])


def render_experiments(api_base):
    st.header("🧪 检索模块消融评估")
    st.caption("先逐一添加变体，再用同一组问题、同一索引快照与相同 top_k 并行比较。")
    try:
        if "ablation_variants" not in st.session_state:
            var_resp = _request(api_base, "GET", "/evaluations/variants")
            st.session_state.ablation_variants = var_resp["variants"]
            if "ablation_last_request" not in st.session_state and var_resp.get("last_request"):
                st.session_state.ablation_last_request = var_resp["last_request"]
        if "ablation_dataset_text" not in st.session_state:
            legacy = st.session_state.get("ablation_dataset", "")
            if isinstance(legacy, str) and legacy.strip():
                st.session_state.ablation_dataset_text = legacy
            else:
                dataset = _request(api_base, "GET", "/evaluations/dataset")["dataset"]
                st.session_state.ablation_dataset_text = json.dumps(dataset, ensure_ascii=False, indent=2)
        elif (
            not st.session_state.get("_ablation_user_edited")
            and '"question_type"' not in st.session_state.ablation_dataset_text
        ):
            parsed_existing, err_existing = _parse_dataset(st.session_state.ablation_dataset_text)
            if not err_existing and len(parsed_existing) >= 30:
                latest_dataset = _request(api_base, "GET", "/evaluations/dataset")["dataset"]
                _set_dataset(json.dumps(latest_dataset, ensure_ascii=False, indent=2), user_edited=False)
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

        dataset, dataset_error = _parse_dataset(st.session_state.ablation_dataset_text)
        if dataset_error:
            st.error(dataset_error)
        _render_dataset_preview(dataset)

        col_reload, _ = st.columns([0.35, 0.65])
        if col_reload.button("恢复项目题集"):
            reloaded_ds = _request(api_base, "GET", "/evaluations/dataset")["dataset"]
            _set_dataset(json.dumps(reloaded_ds, ensure_ascii=False, indent=2), user_edited=False)
            st.rerun()

        with st.expander("⚙️ 高级：导入或手动编辑底层 JSON", expanded=False):
            uploaded = st.file_uploader("导入题集 JSON", type="json", key="eval_dataset_upload")
            if uploaded and st.button("载入上传题集"):
                try:
                    text = uploaded.getvalue().decode("utf-8-sig")
                    parsed, error = _parse_dataset(text)
                    if error:
                        st.error(f"导入失败：{error}")
                    else:
                        _set_dataset(json.dumps(parsed, ensure_ascii=False, indent=2), user_edited=True)
                        st.rerun()
                except UnicodeDecodeError:
                    st.error("题集文件需要使用 UTF-8 编码。")
            st.text_area("题集 JSON", height=300, key="_ablation_dataset_editor", on_change=_sync_dataset_editor)
            st.caption('每题包含 question、ground_truth，可选 question_type（"simple" 或 "complex"）及 relevant_chunk_ids，例如 ["original.pdf::1"]。')
        last_req = st.session_state.get("ablation_last_request") or {}
        all_var_names = [v["name"] for v in variants]
        last_var_names = [n for n in (last_req.get("variant_names") or []) if n in all_var_names]
        default_selected = last_var_names if last_var_names else all_var_names
        if "ablation_selected_variants" not in st.session_state:
            st.session_state["ablation_selected_variants"] = default_selected
        else:
            kept_vars = [n for n in st.session_state["ablation_selected_variants"] if n in all_var_names]
            st.session_state["ablation_selected_variants"] = kept_vars if kept_vars else default_selected
        selected = st.multiselect("本批评估的变体", all_var_names, key="ablation_selected_variants")

        simple_pool = [q for q in dataset if q.get("question_type", "simple") == "simple"]
        complex_pool = [q for q in dataset if q.get("question_type") == "complex"]
        if simple_pool and complex_pool:
            st.caption(f"当前题集共 {len(dataset)} 题：🟢 简单问题 {len(simple_pool)} 题 · 🟣 复杂问题 {len(complex_pool)} 题（拖动滑块可直接拉到最大题数）")
            def_simple = min(int(last_req["simple_count"]), len(simple_pool)) if "simple_count" in last_req else min(3, len(simple_pool))
            def_complex = min(int(last_req["complex_count"]), len(complex_pool)) if "complex_count" in last_req else min(3, len(complex_pool))
            if "ablation_simple_count" not in st.session_state:
                st.session_state["ablation_simple_count"] = def_simple
            else:
                st.session_state["ablation_simple_count"] = min(int(st.session_state["ablation_simple_count"]), len(simple_pool))
            if "ablation_complex_count" not in st.session_state:
                st.session_state["ablation_complex_count"] = def_complex
            else:
                st.session_state["ablation_complex_count"] = min(int(st.session_state["ablation_complex_count"]), len(complex_pool))

            col_s, col_c = st.columns(2)
            simple_count = col_s.slider(
                "本次使用前 N 个简单问题", 0, len(simple_pool),
                disabled=bool(dataset_error), key="ablation_simple_count",
            )
            complex_count = col_c.slider(
                "本次使用前 M 个复杂问题", 0, len(complex_pool),
                disabled=bool(dataset_error), key="ablation_complex_count",
            )
            selected_dataset = simple_pool[:simple_count] + complex_pool[:complex_count]
        else:
            def_count = min(max(1, int(last_req.get("total_count", 3))), max(1, len(dataset)))
            if len(dataset) > 1:
                count = st.slider("本次使用前 N 个问题", 1, len(dataset), def_count, disabled=bool(dataset_error))
            else:
                count = st.number_input("本次使用前 N 个问题", 1, 1, 1, disabled=bool(dataset_error))
            selected_dataset = dataset[:count]

        default_scope_label = "完整 RAG" if last_req.get("scope") == "full" else "仅检索"
        if "ablation_scope_radio" not in st.session_state:
            st.session_state["ablation_scope_radio"] = default_scope_label
        scope = st.radio("评估范围", ["仅检索", "完整 RAG"], horizontal=True, key="ablation_scope_radio")
        st.caption("仅检索：不运行路由、相关性判断、改写及网页回退，比较初次检索结果。完整 RAG：按各变体开关运行完整流程，并额外评分回答忠实度和相关性。")

        if "ablation_use_ragas" not in st.session_state:
            st.session_state["ablation_use_ragas"] = bool(last_req.get("use_ragas", True))
        use_ragas = st.checkbox("启用立即 RAGAs 模型评分（取消勾选则先快速生成回答，后续可一键批量评分）", key="ablation_use_ragas")

        def_conc = min(max(1, int(last_req.get("concurrency", 2))), 5)
        if "ablation_concurrency" not in st.session_state:
            st.session_state["ablation_concurrency"] = def_conc
        concurrency = st.slider("并行变体数", 1, 5, key="ablation_concurrency")

        def_ragas_conc = min(max(1, int(last_req.get("ragas_concurrency", 16) or 16)), 64)
        if "ablation_ragas_concurrency" not in st.session_state:
            st.session_state["ablation_ragas_concurrency"] = def_ragas_conc
        ragas_concurrency = st.slider(
            "🚀 RAGAs API 并发请求数", 1, 64,
            help="控制向 DeepSeek 发起普通 API 评分请求的最大并发数量（与本地检索并发解耦，建议 16~32，最高 64）。",
            key="ablation_ragas_concurrency",
        )

        def_top_k = min(max(1, int(last_req.get("top_k", 5))), 20)
        if "ablation_top_k" not in st.session_state:
            st.session_state["ablation_top_k"] = def_top_k
        top_k = st.slider("本批统一 top_k", 1, 20, key="ablation_top_k")

        def_cand_k = min(max(20, (int(last_req.get("candidate_k", 20)) // 5) * 5), 100)
        if "ablation_candidate_k" not in st.session_state:
            st.session_state["ablation_candidate_k"] = def_cand_k
        candidate_k = st.slider("本批统一候选块数", 20, 100, step=5, key="ablation_candidate_k")

        mode_hint = f"立即执行 RAGAs 评分（API 并发度 {ragas_concurrency}）" if use_ragas else "仅生成检索与回答结果（稍后可单批或多批一键高并发 RAGAs 评分）"
        st.info(f"将评估 {len(selected)} 个变体 × {len(selected_dataset)} 个问题；本地检索/生成并发度 {concurrency}；{mode_hint}。运行记录及完整配置将保存在本机。")
        col_start, col_stop = st.columns([0.7, 0.3])
        if col_start.button("开始批量并行评估", type="primary", use_container_width=True, disabled=bool(dataset_error) or not selected or not selected_dataset or st.session_state.get("ablation_active", False)):
            chosen = [{"name": v["name"], "config": {**v["config"], "top_k": top_k, "candidate_k": candidate_k}}
                      for v in variants if v["name"] in selected]
            scope_val = "retrieval" if scope == "仅检索" else "full"
            job = _request(api_base, "POST", "/evaluations/jobs", json={
                "variants": chosen, "dataset": selected_dataset, "scope": scope_val,
                "concurrency": concurrency, "ragas_concurrency": ragas_concurrency,
                "use_ragas": use_ragas, "defer_ragas": not use_ragas,
            })
            st.session_state.ablation_last_request = {
                "variant_names": list(selected),
                "scope": scope_val,
                "use_ragas": use_ragas,
                "concurrency": concurrency,
                "ragas_concurrency": ragas_concurrency,
                "top_k": top_k,
                "candidate_k": candidate_k,
                "simple_count": sum(1 for q in selected_dataset if q.get("question_type", "simple") == "simple"),
                "complex_count": sum(1 for q in selected_dataset if q.get("question_type") == "complex"),
                "total_count": len(selected_dataset),
            }
            st.session_state.ablation_job = job["id"]
            st.session_state.ablation_active = True
            st.rerun()
        if st.session_state.get("ablation_active", False) and st.session_state.get("ablation_job"):
            if col_stop.button("⏹️ 停止当前评估", key="stop_main_btn", use_container_width=True):
                _request(api_base, "POST", f"/evaluations/jobs/{st.session_state.ablation_job}/cancel")
                st.session_state.ablation_active = False
                st.rerun()
        with st.expander("查看与批量评估历史批次"):
            if st.button("刷新历史批次"):
                st.session_state.ablation_history = _request(api_base, "GET", "/evaluations/jobs")["jobs"]
            history = st.session_state.get("ablation_history", [])
            if history:
                job_label_map = {}
                unscored_ids = []
                for j in history:
                    jid = j["id"]
                    scored_tag = "✅ 已RAGAs评分" if j.get("ragas_scored") else "⏳ 待RAGAs评分"
                    q_cnt = j.get("question_count")
                    q_tag = f"{q_cnt}题 · " if q_cnt else ""
                    v_list = j.get("variant_names") or []
                    v_tag = ("+".join(v_list[:2]) + ("…" if len(v_list) > 2 else "")) if v_list else ""
                    job_label_map[jid] = f"{jid[:10]}… · {scored_tag} · {q_tag}{v_tag or j.get('status', '')}"
                    if not j.get("ragas_scored") and j.get("status") not in ("queued", "running"):
                        unscored_ids.append(jid)
                old_id = st.selectbox("历史批次", [j["id"] for j in history], format_func=lambda x: job_label_map.get(x, x))
                if st.button("打开批次"):
                    st.session_state.ablation_job = old_id
                    st.session_state.ablation_active = True
                    st.rerun()
                st.divider()
                batch_ids = st.multiselect(
                    "选择历史批次进行批量 RAGAs 评分",
                    [j["id"] for j in history],
                    default=unscored_ids,
                    format_func=lambda x: job_label_map.get(x, x),
                    key="ablation_batch_ragas_ids",
                )
                if st.button(
                    f"🚀 批量执行 RAGAs 评分（并发度 {ragas_concurrency}）",
                    disabled=not batch_ids or st.session_state.get("ablation_active", False),
                    key="batch_ragas_eval_btn",
                ):
                    _request(api_base, "POST", "/evaluations/jobs/ragas", json={
                        "job_ids": batch_ids,
                        "ragas_concurrency": ragas_concurrency,
                    })
                    st.session_state.ablation_job = batch_ids[0]
                    st.session_state.ablation_active = True
                    st.rerun()
    except (httpx.HTTPError, ValueError) as exc:
        st.error(f"评估配置或请求失败：{exc}")
    # A draft configuration error must never hide an already-running batch.
    if st.session_state.get("ablation_job"):
        panel = st.fragment(run_every=3 if st.session_state.get("ablation_active") else None)(_job_panel)
        panel(api_base, st.session_state.ablation_job)
