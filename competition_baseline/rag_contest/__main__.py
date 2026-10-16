import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from .data import (audit_and_split, digest, evaluate, extract_documents, file_hash, load_gold,
                   load_questions, write_csv, write_json)
from .models import MODELS, environment, offline, prepare, verify_assets
from .pipeline import Config, ResourceMonitor, run_questions, source_fingerprint
from .retrieval import Retriever, WordCodec
from .index_cache import cached_documents, cached_retriever


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def retrieval_report(args):
    docs = extract_documents(args.docs)
    questions, gold = load_questions(args.questions), load_gold(args.gold)
    if set(r["question_id"] for r in questions) != set(r["question_id"] for r in gold):
        raise ValueError("Question/gold IDs differ")
    gold = {r["question_id"]: r for r in gold}
    retriever = Retriever(docs, WordCodec())
    result = []
    for row in questions:
        ids = [c["document_id"] for c in retriever.search(row["question_text"])]
        expected = gold[row["question_id"]]["document_id"]
        result.append({"question_id": row["question_id"], "expected": expected, "retrieved": ids,
                       "top1": ids[0] == expected, "top3": expected in ids})
    report = {"method": "BM25 only, word-window diagnostic (not the hybrid model pipeline)",
              "count": len(result), "documents": len(docs), "chunks": len(retriever.chunks),
              "top1_accuracy": sum(r["top1"] for r in result) / len(result),
              "top3_recall": sum(r["top3"] for r in result) / len(result), "details": result}
    write_json(args.output, report)
    return {k: v for k, v in report.items() if k != "details"}


def run(args):
    config = Config(**read_json(args.config)) if args.config else Config()
    if config.device != 'cpu' and args.lock:
        raise ValueError('Competition lock requires CPU; CUDA is development only')
    output = Path(args.output)
    if (output / "answer.csv").exists() or (args.fresh and (output / "checkpoint.json").exists()):
        raise ValueError("Use a new output directory for a fresh or previously completed run")
    output.mkdir(parents=True, exist_ok=True)
    manifest = None
    cache_report = {}
    cache_dir = Path(args.assets).parent.parent / "index_cache"
    history_path = output / 'metrics_history.json'
    history = read_json(history_path) if history_path.exists() else []
    prior_seconds = sum(h['elapsed_seconds'] for h in history)
    setup_seconds = next((h['setup_seconds'] for h in history if h.get('setup_seconds') is not None), args.setup_seconds)
    available = min(config.max_seconds - prior_seconds,
                    config.total_seconds - (setup_seconds or 0) - prior_seconds)
    if available <= 0:
        raise ValueError('Run has exhausted its cumulative time budget; inspect saved results before starting another run')
    with ResourceMonitor(available, config.max_ram_gb) as monitor:
        try:
            print("[1/5] ตรวจไฟล์โมเดลและ hash...", flush=True)
            manifest = verify_assets(args.assets)
            env = environment()
            if args.lock:
                lock = read_json(args.lock)
                if (lock["config"] != asdict(config) or lock["model_digest"] != digest(manifest)
                        or lock["sources"] != source_fingerprint()
                        or lock["packages"] != env["packages"] or lock["python"] != env["python"]
                        or lock["resolved_dependencies"] != env["resolved_dependencies"]):
                    raise ValueError("Competition lock does not match code/config/model/dependencies")
            questions = load_questions(args.questions)
            if args.expected_count and len(questions) != args.expected_count:
                raise ValueError(f"Expected {args.expected_count} questions, found {len(questions)}")
            if args.expected_count == 100 and {r["question_id"] for r in questions} != {str(i) for i in range(1, 101)}:
                raise ValueError("Competition questions must have IDs 1 through 100")
            print("[2/5] ตรวจ PDF และโหลดข้อความที่บันทึกไว้...", flush=True)
            docs, cache_report["documents"] = cached_documents(args.docs, cache_dir)
            print("ข้อความ PDF: " + ("ใช้ cache" if cache_report["documents"]["hit"] else "สร้างใหม่"), flush=True)
            print(f"อ่านได้ {len(docs)} เอกสาร; คำถามรอบนี้ {len(questions)} ข้อ", flush=True)
            write_csv(output / "questions_used.csv", questions, ["question_id", "question_text"])
            provenance = {"documents": [{k: d[k] for k in ("document_id", "filename", "sha256")} for d in docs],
                          "model_digest": digest(manifest), "model_name": manifest["model_name"], "environment": env,
                          "sources": source_fingerprint()}
            with offline():
                from .models import Generator
                from .retrieval import DenseEncoder
                print("[3/5] โหลดโมเดลตอบคำถามบน CPU...", flush=True)
                generator = Generator(args.assets, manifest, config.threads, config.context_tokens, config.seed, device=config.device)
                monitor.check()
                print("[4/5] เตรียมโมเดลค้นหาและสร้างดัชนี...", flush=True)
                dense = DenseEncoder(Path(args.assets) / "embedding", config.threads) if config.retrieval == "hybrid" else None
                retriever, cache_report["index"] = cached_retriever(
                    docs, generator, dense, config, cache_dir,
                    {"model_digest": digest(manifest), "environment": env})
                print("ดัชนีค้นหา: " + ("ใช้ cache" if cache_report["index"]["hit"] else "สร้างใหม่"), flush=True)
                monitor.check()
                initialization_seconds = time.perf_counter() - monitor.start
                inference_start = time.perf_counter()
                print("[5/5] เริ่มตอบคำถาม...", flush=True)
                try:
                    result = run_questions(questions, retriever, generator, config, output, provenance,
                                           resume=not args.fresh, check=monitor.check)
                except TimeoutError:
                    from .pipeline import timeout_report
                    result = timeout_report(questions, output, config, provenance)
                    print("หมดงบเวลา: เก็บผลบางส่วนเพื่อวัดคะแนน ยังไม่มีไฟล์ส่งครบ", flush=True)
                inference_seconds = time.perf_counter() - inference_start
            return result
        finally:
            metrics = monitor.report()
            metrics['device'] = config.device
            metrics['development_only'] = config.device != 'cpu'
            from .runtime import cpu_report
            metrics['runtime'] = cpu_report(config.threads)
            metrics['assets_path'] = str(args.assets)
            metrics['initialization_seconds'] = locals().get('initialization_seconds')
            metrics['answering_seconds'] = locals().get('inference_seconds')
            metrics["index_cache"] = cache_report
            metrics["fresh"] = bool(args.fresh)
            history.append({'elapsed_seconds': metrics['elapsed_seconds'], 'fresh': bool(args.fresh),
                            'setup_seconds': args.setup_seconds})
            write_json(history_path, history)
            metrics['session_count'] = len(history)
            metrics['cumulative_run_seconds'] = sum(h['elapsed_seconds'] for h in history)
            metrics['cumulative_total_seconds'] = (setup_seconds + metrics['cumulative_run_seconds']
                                                   if setup_seconds is not None else None)
            metrics['total_budget_seconds'] = config.total_seconds
            metrics['timing_note'] = 'elapsed_seconds is this invocation only; cumulative time excludes unrecorded earlier invocations and pauses.'
            metrics["setup_seconds"] = args.setup_seconds
            metrics["cold_start_total_seconds"] = (args.setup_seconds + metrics["elapsed_seconds"]
                                                    if args.setup_seconds is not None else None)
            metrics["model_name"] = manifest["model_name"] if manifest else None
            metrics["question_count"] = len(questions) if "questions" in locals() else None
            metrics["projected_100_seconds_excluding_setup"] = (
                initialization_seconds + inference_seconds * 100 / len(questions)
                if "inference_seconds" in locals() and args.fresh and result.get('complete') else None)
            write_json(output / "metrics.json", metrics)


def score_run(args):
    questions, gold = load_questions(args.questions), load_gold(args.gold)
    run_dir = Path(args.run)
    status = read_json(run_dir / "status.json")
    traces = {r["question_id"]: r for r in read_json(run_dir / "diagnostics.json")}
    if set(traces) != {r["question_id"] for r in questions}:
        raise ValueError("Run questions do not match evaluation questions")
    predictions = [{"question_id": q["question_id"],
                    "answer_text": traces[q["question_id"]].get("answer_text", ""),
                    "document_id": traces[q["question_id"]].get("document_id", "")}
                   for q in questions]
    report = evaluate(questions, gold, predictions)
    # Unresolved cases receive zero for the answer, even if empty short text has partial LD credit.
    unresolved = set(status["unresolved_ids"])
    for d in report["details"]:
        if d["question_id"] in unresolved:
            d["answer_score"] = d["document_score"] = 0.0
    report["score_out_of_200"] = 100 * sum(d["answer_score"] + d["document_score"] for d in report["details"]) / len(questions)
    for kind in report["answer_score_by_type"]:
        subset = [d for d in report["details"] if d["kind"] == kind]
        report["answer_score_by_type"][kind] = sum(d["answer_score"] for d in subset) / len(subset)
    report.update(complete=status["complete"], evaluation_set=args.label,
                  evaluation_digest=digest([questions, gold]), run_fingerprint=status["fingerprint"],
                  metrics=read_json(run_dir / "metrics.json"))
    report['completion_meaning'] = 'All answers accepted by inferred types; not a gold-format or correctness guarantee'
    report['gold_format_valid'] = report['format_valid_fraction'] == 1.0
    report['type_mismatch_ids'] = [d['question_id'] for d in report['details']
                                   if traces[d['question_id']].get('kind') is not None and traces[d['question_id']]['kind'] != d['kind']]
    report['unclassified_ids'] = [q['question_id'] for q in questions if traces[q['question_id']].get('kind') is None]
    report['progress_counts'] = {
        'accepted_by_pipeline': sum(t.get('status') == 'ok' for t in traces.values()),
        'awaiting_retry': sum(bool(t.get('retry_pending')) for t in traces.values()),
        'without_saved_calls': sum(not t.get('calls') for t in traces.values()),
        'gold_format_valid': sum(d['format_valid'] for d in report['details']),
    }
    write_json(run_dir / "score.json", report)
    if args.label == "holdout":
        write_json(run_dir / "HOLDOUT_OPENED.json", {"warning": "This set has now been inspected. Further tuning on it makes it a development set."})
    return {k: v for k, v in report.items() if k != "details"}


def compare(args):
    entries = []
    evaluation_digest = None
    for directory in args.runs:
        path = Path(directory)
        score = read_json(path / "score.json")
        if score["evaluation_set"] != "dev":
            raise ValueError("Select candidates using dev scores only")
        if evaluation_digest and score["evaluation_digest"] != evaluation_digest:
            raise ValueError("Cannot compare scores from different datasets")
        evaluation_digest = score["evaluation_digest"]
        metrics = score["metrics"]
        elapsed = metrics["projected_100_seconds_excluding_setup"]
        eligible = (score["complete"] and score["format_valid_fraction"] == 1 and elapsed is not None
                    and elapsed <= 5400 and metrics["peak_process_ram_gb"] <= 12.7
                    and metrics["peak_system_used_gb"] <= 12.7)
        entries.append({"run": str(path), "model": metrics["model_name"], "score": score["score_out_of_200"],
                        "eligible": eligible, "projected_100_seconds_excluding_setup": elapsed})
    entries.sort(key=lambda r: (not r["eligible"], -r["score"], r["projected_100_seconds_excluding_setup"] or float("inf")))
    report = {"winner": next((e["run"] for e in entries if e["eligible"]), None), "candidates": entries,
              "note": "Provisional projection only, excluding installation/download. Final lock requires measured clean 100-question rehearsal and uncached repeat verification."}
    write_json(args.output, report)
    return report


def freeze(args):
    run_path = Path(args.run)
    score, status, metrics = (read_json(run_path / n) for n in ("score.json", "status.json", "metrics.json"))
    if status['config'].get('device', 'cpu') != 'cpu':
        raise ValueError('Cannot freeze a GPU development run for competition')
    if not score["complete"] or score["evaluation_set"] != "dev":
        raise ValueError("Lock requires a complete scored development run")
    repeat = read_json(args.repeat)
    rehearsal = read_json(Path(args.rehearsal) / "metrics.json")
    rehearsal_status = read_json(Path(args.rehearsal) / "status.json")
    rehearsal_rows = load_questions(Path(args.rehearsal) / "questions_used.csv")
    if not repeat["identical"] or len(rehearsal_rows) != 100 or not rehearsal_status["complete"]:
        raise ValueError("Lock requires identical uncached runs and a complete 100-question rehearsal")
    if repeat["fingerprint"] != rehearsal_status["fingerprint"]:
        raise ValueError("Repeat verification must cover the exact 100-question rehearsal")
    if (rehearsal["cold_start_total_seconds"] is None or rehearsal["cold_start_total_seconds"] > 5400
            or rehearsal["peak_system_used_gb"] > 12.7 or rehearsal["peak_process_ram_gb"] > 12.7):
        raise ValueError("Rehearsal failed cold-start time/RAM limits")
    manifest = verify_assets(args.assets)
    if digest(manifest) != status["provenance"]["model_digest"]:
        raise ValueError("Assets do not match the scored run")
    if (status["config"] != rehearsal_status["config"]
            or status["provenance"]["model_digest"] != rehearsal_status["provenance"]["model_digest"]):
        raise ValueError("Rehearsal does not use the selected configuration/model")
    if repeat["config"] != status["config"] or repeat["model_digest"] != status["provenance"]["model_digest"]:
        raise ValueError("Repeat verification uses a different configuration/model")
    env = environment()
    if env != status["provenance"]["environment"] or env != rehearsal_status["provenance"]["environment"]:
        raise ValueError("Freeze must run in the same environment as the validated runs")
    if any(s["provenance"]["sources"] != source_fingerprint() for s in (status, rehearsal_status)):
        raise ValueError("Source code changed after the validated runs")
    write_json(args.output, {"config": status["config"], "model_digest": digest(manifest),
               "sources": source_fingerprint(), "packages": env["packages"], "python": env["python"],
               "resolved_dependencies": env["resolved_dependencies"],
               "model_name": manifest["model_name"], "development_run": str(run_path),
               "rehearsal": str(args.rehearsal), "repeat_verification": repeat})
    return {"lock": str(args.output), "model": manifest["model_name"]}


def verify_repeat(args):
    a, b = Path(args.first), Path(args.second)
    sa, sb = read_json(a / "status.json"), read_json(b / "status.json")
    ma, mb = read_json(a / "metrics.json"), read_json(b / "metrics.json")
    if not ma.get("fresh") or not mb.get("fresh"):
        raise ValueError("Both runs must use --fresh and new output directories")
    if not sa["complete"] or not sb["complete"] or sa["fingerprint"] != sb["fingerprint"]:
        raise ValueError("Repeat runs must be complete with identical fingerprints")
    result = {"identical": file_hash(a / "answer.csv") == file_hash(b / "answer.csv"),
              "fingerprint": sa["fingerprint"], "first": str(a), "second": str(b),
              "answer_sha256": file_hash(a / "answer.csv"), "config": sa["config"],
              "model_digest": sa["provenance"]["model_digest"]}
    write_json(args.output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description="CPU-only RAG contest tools")
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit")
    audit.add_argument("--questions", required=True)
    audit.add_argument("--gold", required=True)
    audit.add_argument("--output", default="outputs/data")
    retrieve = sub.add_parser("retrieval-check")
    for flag in ("docs", "questions", "gold", "output"):
        retrieve.add_argument("--" + flag, required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--model", choices=MODELS, required=True)
    prep.add_argument("--assets", required=True)
    execute = sub.add_parser("run")
    for flag in ("docs", "questions", "assets", "output"):
        execute.add_argument("--" + flag, required=True)
    execute.add_argument("--config")
    execute.add_argument("--lock")
    execute.add_argument("--fresh", action="store_true")
    execute.add_argument("--expected-count", type=int)
    execute.add_argument("--setup-seconds", type=float, help="Measured install + prepare time for this clean rehearsal")
    score = sub.add_parser("score")
    for flag in ("questions", "gold", "run"):
        score.add_argument("--" + flag, required=True)
    score.add_argument("--label", required=True, choices=("dev", "holdout"))
    comparison = sub.add_parser("compare")
    comparison.add_argument("--runs", nargs="+", required=True)
    comparison.add_argument("--output", default="outputs/comparison.json")
    repeat = sub.add_parser("verify-repeat")
    for flag in ("first", "second", "output"):
        repeat.add_argument("--" + flag, required=True)
    lock = sub.add_parser("freeze")
    for flag in ("run", "assets", "repeat", "rehearsal", "output"):
        lock.add_argument("--" + flag, required=True)
    export = sub.add_parser("export-competition")
    export.add_argument("--lock", required=True)
    export.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "audit":
        result = audit_and_split(load_questions(args.questions), load_gold(args.gold), args.output)
    elif args.command == "retrieval-check":
        result = retrieval_report(args)
    elif args.command == "prepare":
        manifest = prepare(args.model, args.assets)
        result = {k: manifest[k] for k in ("model_name", "revisions", "prepare_seconds")}
    elif args.command == "run":
        result = run(args)
    elif args.command == "score":
        result = score_run(args)
    elif args.command == "compare":
        result = compare(args)
    elif args.command == "verify-repeat":
        result = verify_repeat(args)
    elif args.command == "export-competition":
        from .competition import export_notebook
        result = export_notebook(args.lock, args.output)
    else:
        result = freeze(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.command == "run" and not result["complete"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
