"""Bounded read-only Forge dry-run transport comparison; never commits entries.

Every POST hardcodes dry_run=true. Certificates and hostname checks remain enabled.
Only exception classes, SSL reason codes, HTTP metadata and counts are recorded.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import ssl
import time
from urllib.parse import quote

import certifi
import httpx
from dotenv import load_dotenv

from bufogen.integrations.forge import reviewed_entries
from bufogen.settings import ROOT
from bufogen.settings import required


def error_chain(error):
    values, seen = [], set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        values.append({
            "type": type(error).__module__ + "." + type(error).__name__,
            "ssl_reason": getattr(error, "reason", None) if isinstance(error, ssl.SSLError) else None,
            "errno": getattr(error, "errno", None),
        })
        error = error.__cause__ or error.__context__
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=ROOT / "runs/forge-iteration2/training/e1/forge-state.json")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/forge-iteration2/diagnostics/upload-transport.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve existing diagnostic evidence; choose a new output")
    load_dotenv(ROOT / ".env")
    state = json.loads(args.state.read_text())
    source = Path(state["source_path"]).read_bytes()
    if hashlib.sha256(source).hexdigest() != state["source_sha256"]:
        raise ValueError("Frozen training source changed")
    entries = reviewed_entries([json.loads(line) for line in source.splitlines() if line.strip()],
                               heldout_ids=state["heldout_ids"], heldout_families=state["heldout_families"])
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if hashlib.sha256(canonical).hexdigest() != state["upload_sha256"]:
        raise ValueError("Mapped upload source changed")
    h2_available = importlib.util.find_spec("h2") is not None
    report = {
        "schema_version": 1,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "dry_run_only": True,
        "certificate_verification": "CERT_REQUIRED, hostname check, certifi CA bundle",
        "openssl_version": ssl.OPENSSL_VERSION,
        "h2_available": h2_available,
        "source_sha256": state["source_sha256"],
        "upload_sha256": state["upload_sha256"],
        "total_source_entries": len(entries),
        "cases": [],
    }
    endpoint = (state["base_url"].rstrip("/") + "/projects/" + quote(state["project_alias"], safe="")
                + "/datasets/" + quote(state["dataset_id"], safe="") + "/entries")
    headers = {"Authorization": "Bearer " + required("WANDB_API_KEY"), "Wandb-Entity": state["entity"]}
    variants = [("http1-default-small1", 1, False, False),
                ("http1-default-small10", 10, False, False),
                ("http1-default-full", len(entries), False, False),
                ("http2-default-full", len(entries), True, False),
                ("http1-tls12-full", len(entries), False, True),
                ("http2-tls12-full", len(entries), True, True)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.output)
    save()
    for name, count, http2, tls12 in variants:
        item = {"name": name, "entry_count": count, "requested_http2": http2,
                "tls12_only": tls12, "started_at": datetime.now(timezone.utc).isoformat()}
        if http2 and not h2_available:
            item["status"] = "skipped_h2_not_installed"
            report["cases"].append(item)
            save()
            print(json.dumps(item), flush=True)
            continue
        context = ssl.create_default_context(cafile=certifi.where())
        if tls12:
            context.minimum_version = ssl.TLSVersion.TLSv1_2
            context.maximum_version = ssl.TLSVersion.TLSv1_2
        assert context.verify_mode == ssl.CERT_REQUIRED and context.check_hostname
        start = time.monotonic()
        try:
            with httpx.Client(verify=context, http2=http2, headers=headers,
                              timeout=httpx.Timeout(30, connect=10),
                              limits=httpx.Limits(max_keepalive_connections=0)) as client:
                request = client.build_request("POST", endpoint,
                                               params={"mode": "append", "dry_run": "true"},
                                               json=entries[:count])
                item["request_body_bytes"] = len(request.content)
                response = client.send(request)
                item.update(http_status=response.status_code, http_version=response.http_version)
                stream = response.extensions.get("network_stream")
                ssl_object = stream.get_extra_info("ssl_object") if stream else None
                item["negotiated_tls"] = ssl_object.version() if ssl_object else None
                if response.is_success:
                    result = response.json()
                    item.update(status="validated" if result.get("dry_run") is True else "unexpected_dry_run_response",
                                dry_run=result.get("dry_run"), revision=result.get("revision"),
                                created=result.get("created"), updated=result.get("updated"),
                                entry_counts=result.get("entry_counts"))
                else:
                    item["status"] = "http_error"
        except Exception as exc:
            item.update(status="transport_error", exception_chain=error_chain(exc))
        item["elapsed_s"] = round(time.monotonic() - start, 3)
        report["cases"].append(item)
        save()
        print(json.dumps(item), flush=True)
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    save()


if __name__ == "__main__":
    main()
