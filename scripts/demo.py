"""Repeatable end-to-end demonstration of the whole workflow.

    python -m scripts.demo                       # against http://127.0.0.1:8000
    python -m scripts.demo --base-url http://127.0.0.1:8099
    python -m scripts.demo --username alex       # use an existing moderator

Walks the complete path a reporter and a moderator actually take, printing
what each of them sees at every step, and asserting the privacy guarantees as
it goes. It exits non-zero if any of them fails, so it is a smoke test as well
as a demonstration.

What it needs
-------------
A running API and a moderator account. The moderator's password is read from
``WHISTLEDROP_DEMO_PASSWORD`` or prompted for without echo — never passed as an
argument, for the same reason the seeding CLI refuses one.

What it demonstrates
--------------------
1.  the service is up
2.  a report is filed with no account and no identifying field
3.  a case code comes back, once
4.  the reporter can track the case with it
5.  a moderator logs in
6.  the queue, filtered
7.  the report in full, including the AI suggestion
8.  that the suggestion did **not** change the official category
9.  SUBMITTED -> UNDER_REVIEW with an internal note
10. that the reporter cannot see that note
11. UNDER_REVIEW -> RESOLVED with a published note
12. that the reporter can see this one
13. that an illegal transition is refused
14. that no moderator identity, internal note, AI data or database id ever
    reaches the reporter
"""

import argparse
import getpass
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any

DEFAULT_BASE_URL = "http://127.0.0.1:8000"

# Deliberately synthetic. Never demonstrate with a real report.
DEMO_REPORT = {
    "category": "OTHER",
    "description": (
        "Demo report, not a real one. Production database credentials are being "
        "shared in a public chat channel and anyone in the company can read them."
    ),
}
INTERNAL_NOTE = "INTERNAL DEMO NOTE: the reporter must never see this line."
PUBLISHED_NOTE = "The credentials have been rotated and access has been revoked."

GREEN, RED, DIM, BOLD, RESET = "\033[32m", "\033[31m", "\033[2m", "\033[1m", "\033[0m"

failures: list[str] = []


def check(label: str, condition: bool) -> None:
    mark = f"{GREEN}PASS{RESET}" if condition else f"{RED}FAIL{RESET}"
    print(f"    [{mark}] {label}")
    if not condition:
        failures.append(label)


def step(number: int, title: str) -> None:
    print(f"\n{BOLD}{number:2}. {title}{RESET}")


class Api:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.token: str | None = None

    def __call__(
        self, method: str, path: str, payload: Any = None, *, auth: bool = False
    ) -> tuple[int, Any]:
        headers = {"Content-Type": "application/json"}
        if auth and self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        data = json.dumps(payload).encode() if payload is not None else None
        # S310 flags urlopen on a URL it cannot prove is http(s). The base URL
        # is an operator-supplied argument to a local demo script, not request
        # input, and the scheme is checked below.
        if not self.base_url.startswith(("http://", "https://")):
            raise SystemExit(f"--base-url must be http(s), got {self.base_url!r}")
        request = urllib.request.Request(  # noqa: S310
            self.base_url + path, data=data, method=method, headers=headers
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:  # noqa: S310
                body = response.read()
                return response.status, (json.loads(body) if body else None)
        except urllib.error.HTTPError as exc:
            body = exc.read()
            return exc.code, (json.loads(body) if body else None)
        except urllib.error.URLError as exc:
            print(f"\n{RED}Cannot reach {self.base_url}: {exc.reason}{RESET}")
            print("Start the API first:  uvicorn app.main:app --reload")
            raise SystemExit(2) from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.demo",
        description="End-to-end demonstration of the WhistleDrop workflow.",
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--username", default="demo.moderator")
    args = parser.parse_args(argv)

    # Never an argument: it would land in shell history and the process list.
    password = os.environ.get("WHISTLEDROP_DEMO_PASSWORD") or getpass.getpass(
        f"Password for moderator {args.username!r}: "
    )

    api = Api(args.base_url)
    print(f"{BOLD}WhistleDrop end-to-end demonstration{RESET}")
    print(f"{DIM}against {api.base_url}{RESET}")

    # -- 1 ------------------------------------------------------------------
    step(1, "The service is up")
    status, health = api("GET", "/api/v1/health")
    check(f"GET /api/v1/health -> {status}", status == 200)
    print(
        f"    {DIM}service={health['service']} version={health['version']} "
        f"env={health['environment']}{RESET}"
    )

    # -- 2, 3 ---------------------------------------------------------------
    step(2, "A reporter files a report — no account, no identifying field")
    status, submission = api("POST", "/api/v1/reports", DEMO_REPORT)
    check(f"POST /api/v1/reports -> {status}", status == 201)
    if status != 201:
        print(f"    {RED}{json.dumps(submission)}{RESET}")
        return 1

    case_code = submission["case_code"]
    print(
        f"    {DIM}case code: {BOLD}{case_code}{RESET}{DIM}  (shown once, never recoverable){RESET}"
    )
    check(
        "the response carries only the four documented fields",
        set(submission) == {"case_code", "status", "submitted_at", "message"},
    )
    check("the new report is SUBMITTED", submission["status"] == "SUBMITTED")

    step(3, "An identifying field would be refused, not ignored")
    status, _ = api("POST", "/api/v1/reports", {**DEMO_REPORT, "email": "me@example.com"})
    check(f"a body carrying `email` -> {status}", status == 422)

    # -- 4 ------------------------------------------------------------------
    step(4, "The reporter tracks the case with that code")
    status, tracked = api("POST", "/api/v1/cases/lookup", {"case_code": case_code})
    check(f"POST /api/v1/cases/lookup -> {status}", status == 200)
    print(f"    {DIM}status={tracked['status']} category={tracked['category']}{RESET}")
    for entry in tracked["timeline"]:
        print(f"    {DIM}  - {entry['status']}: {entry['note']}{RESET}")
    check(
        "an unknown code is refused",
        api("POST", "/api/v1/cases/lookup", {"case_code": "WD-00000-00000-00000-00000"})[0] == 404,
    )

    # -- 5 ------------------------------------------------------------------
    step(5, "A moderator logs in")
    status, token = api(
        "POST", "/api/v1/auth/login", {"username": args.username, "password": password}
    )
    if status != 200:
        print(f"    {RED}Login failed ({status}). Create a moderator first:{RESET}")
        print("      python -m scripts.create_moderator --username demo.moderator")
        return 1
    api.token = token["access_token"]
    check("POST /api/v1/auth/login -> 200", True)
    print(f"    {DIM}token_type={token['token_type']} expires_in={token['expires_in']}s{RESET}")
    check(
        "the moderation queue is closed without a token",
        Api(api.base_url)("GET", "/api/v1/moderation/reports")[0] == 401,
    )

    # -- 6 ------------------------------------------------------------------
    step(6, "The moderation queue, filtered")
    status, queue = api("GET", "/api/v1/moderation/reports?status=SUBMITTED&page_size=5", auth=True)
    check(f"GET /api/v1/moderation/reports -> {status}", status == 200)
    print(
        f"    {DIM}{queue['page']['total_items']} SUBMITTED report(s), "
        f"page {queue['page']['page']} of {queue['page']['total_pages']}{RESET}"
    )
    report_id = queue["items"][0]["id"]

    # -- 7, 8 ---------------------------------------------------------------
    step(7, "The report in full, with the AI suggestion beside the decision")
    status, detail = api("GET", f"/api/v1/moderation/reports/{report_id}", auth=True)
    check(f"GET .../{{report_id}} -> {status}", status == 200)
    triage = detail["triage"]
    print(
        f"    {DIM}OFFICIAL category : {BOLD}{detail['category']}{RESET}"
        f"{DIM}   <- the human's{RESET}"
    )
    if triage:
        print(
            f"    {DIM}AI suggestion     : {triage['suggested_category']} "
            f"(model probability {triage['category_confidence']})   <- advisory{RESET}"
        )
        print(
            f"    {DIM}AI priority       : {triage['suggested_priority']}"
            f"  (heuristic, not a model){RESET}"
        )
        print(
            f"    {DIM}model version     : {triage['model_version']}"
            f"  [trained on SYNTHETIC data]{RESET}"
        )
        check(
            "the AI did not change the official category",
            detail["category"] == DEMO_REPORT["category"],
        )
        check("the AI did not change the status", detail["status"] == "SUBMITTED")
        check("the AI added no timeline entry", len(detail["timeline"]) == 1)
    else:
        print(f"    {DIM}no triage row (model unavailable — reports are unaffected){RESET}")
    check("case_code_hash is never returned", "case_code_hash" not in json.dumps(detail))

    # -- 8, 9 --------------------------------------------------------------
    step(8, "SUBMITTED -> UNDER_REVIEW, with an INTERNAL note")
    status, _ = api(
        "PATCH",
        f"/api/v1/moderation/reports/{report_id}/status",
        {"status": "UNDER_REVIEW", "note": INTERNAL_NOTE, "visible_to_reporter": False},
        auth=True,
    )
    check(f"PATCH .../status -> {status}", status == 200)

    step(9, "The reporter sees the new status but NOT the internal note")
    status, tracked = api("POST", "/api/v1/cases/lookup", {"case_code": case_code})
    check("the reporter sees status UNDER_REVIEW", tracked["status"] == "UNDER_REVIEW")
    check("the internal note is not disclosed", INTERNAL_NOTE not in json.dumps(tracked))
    check("the timeline still shows only the published entry", len(tracked["timeline"]) == 1)

    # -- 10, 11 -------------------------------------------------------------
    step(10, "UNDER_REVIEW -> RESOLVED, with a note the reporter SHOULD see")
    status, _ = api(
        "PATCH",
        f"/api/v1/moderation/reports/{report_id}/status",
        {"status": "RESOLVED", "note": PUBLISHED_NOTE, "visible_to_reporter": True},
        auth=True,
    )
    check(f"PATCH .../status -> {status}", status == 200)

    step(11, "The reporter sees the resolution")
    status, final = api("POST", "/api/v1/cases/lookup", {"case_code": case_code})
    check("the reporter sees status RESOLVED", final["status"] == "RESOLVED")
    for entry in final["timeline"]:
        print(f"    {DIM}  - {entry['status']}: {entry['note']}{RESET}")
    check(
        "the published note is visible",
        any(PUBLISHED_NOTE == entry["note"] for entry in final["timeline"]),
    )
    check("the internal note is still hidden", INTERNAL_NOTE not in json.dumps(final))

    # -- 12 -----------------------------------------------------------------
    step(12, "An illegal transition is refused")
    status, refused = api(
        "PATCH",
        f"/api/v1/moderation/reports/{report_id}/status",
        {"status": "SUBMITTED"},
        auth=True,
    )
    check(f"RESOLVED -> SUBMITTED -> {status} {refused['error']['code']}", status == 409)

    # -- 13 -----------------------------------------------------------------
    step(13, "Nothing internal ever reached the reporter")
    blob = json.dumps(final)
    check("no moderator username", args.username not in blob)
    # The bare word "moderator" legitimately appears in a published note
    # ("queued for review by a moderator") — that is the role, not an identity.
    # What must be absent is the field and any actual identifier.
    check("no moderator_id field", "moderator_id" not in blob)
    check(
        "no moderator identity in any timeline entry",
        all(set(entry) == {"status", "note", "occurred_at"} for entry in final["timeline"]),
    )
    check("no report database id", report_id not in blob)
    check(
        "no AI triage data",
        not any(k in blob for k in ("suggested_", "confidence", "model_version", "triage")),
    )
    check("no case_code_hash", "hash" not in blob.lower())
    check(
        "exactly the five documented fields",
        set(final) == {"status", "category", "submitted_at", "last_updated_at", "timeline"},
    )

    print()
    if failures:
        print(f"{RED}{BOLD}{len(failures)} check(s) FAILED:{RESET}")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print(f"{GREEN}{BOLD}All checks passed.{RESET}")
    print(f"{DIM}Reminder: the triage model is trained on synthetic data. Its suggestions")
    print(f"are advisory and its metrics do not establish real-world accuracy.{RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
