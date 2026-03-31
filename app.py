from __future__ import annotations

import csv
import json
import re
import uuid
from pathlib import Path
from typing import List, Tuple, Optional
import difflib

import fitz
from flask import Flask, render_template, request, redirect, url_for, session, send_file, flash
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "wage-highlighter-local-v3"

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
STATE_DIR = BASE_DIR / "state"
for p in (UPLOAD_DIR, OUTPUT_DIR, STATE_DIR):
    p.mkdir(exist_ok=True)

CONFIG = {
    "min_wage_amount": 6000.0,
    "max_wage_amount": 45000.0,
    "min_match_score": 0.45,
    "review_candidates": 20,
}

IGNORE_WORDS = {
    "HOUSE", "KEEPER", "GRAND", "TOTAL", "TOT", "FOR", "PARTNER", "SIGNATURE",
    "DETAILS", "SHREE", "SAI", "ENGINEERING", "LOTHPUR", "AMRELI", "GUJARAT",
    "PAGE", "OF", "BONUS", "LWP", "TECHNICIAN", "INCHARGE", "FILLER", "WD",
    "SL", "WO", "PD", "PL", "PH", "CL", "CONTRACTOR", "UNIT", "SITE",
    "DEPARTMENT", "NAME", "ESTABLISHMENT", "OT/PIB", "HRA", "CONV", "DA",
    "BASIC", "CONSOL", "EARNINGS", "SALARY", "DEDUCTION", "NET", "PAYABLE",
    "MONTH", "DAYS", "WORKER", "STAFF", "EMPLOYEE", "OPERATOR", "OPERATORS",
    "LEAVE", "BANK", "TOTAL", "DIFF", "ADV", "LOAN", "GROSS", "RATE", "WAGES",
    "ATTENDENCE", "DESIGNATION", "PUNE", "BOOSTER", "TEAM", "FORM", "RULE"
}
BAN_TOKENS = {
    "EXP", "DELTA", "LTD", "LLP", "PVT", "LIMITED", "CO", "CORP", "INC", "BIKE", "SHOES",
    "BANK", "INDUSIND", "PUNJAB", "NATIONAL", "CANARA", "AXIS", "HDFC", "ICICI",
    "IDFC", "SBI", "STATE", "OF", "UCO", "IOB", "INDIAN", "FEDERAL", "KOTAK",
    "OVERSEAS", "THE", "MAHINDRA", "LEAVE", "BONUS", "TOTAL"
}
BANK_PREFIXES = {
    "BANK", "INDIAN", "INDUSIND", "PUNJAB", "NATIONAL", "CANARA", "AXIS", "HDFC", "ICICI",
    "IDFC", "SBI", "STATE", "OF", "UCO", "IOB", "FEDERAL", "KOTAK",
    "OVERSEAS", "THE", "MAHINDRA",
}
RECORD_PREFIX_RE = re.compile(
    r"^(?:AR000|SR\.|EMP|UAN|PF|TOTAL|P\s|SL\s|WD\s|WO\s|PD\s|PL\s|CL\s|BONUS|LWP|"
    r"TECHNICIAN|INCHARGE|FILLER|OT/PIB|DEPARTMENT|CONTRACTOR|UNIT|SITE|NAME|ESTABLISHMENT)",
    re.IGNORECASE,
)


def normalize_name(name: str) -> str:
    name = re.sub(r"[^A-Z ]", " ", name.upper())
    return " ".join(token for token in name.split() if len(token) >= 2 and token not in IGNORE_WORDS)


def is_probable_name_line(line: str) -> bool:
    upper = re.sub(r"\s+", " ", line.upper()).strip()
    if not upper:
        return False
    if any(ch.isdigit() for ch in upper):
        return False

    tokens = re.findall(r"[A-Z]+", upper)
    if not tokens:
        return False

    # Allow person-name patterns with initials like "P BABU" or
    # "S THOMAS PURUSOTHANAN" even though they start with a single letter.
    looks_like_initial_name = (
        len(tokens) >= 2
        and len(tokens[0]) == 1
        and all(tok.isalpha() for tok in tokens[1:])
    )

    if RECORD_PREFIX_RE.search(upper) and not looks_like_initial_name:
        return False

    if len(tokens) == 1 and tokens[0] in IGNORE_WORDS:
        return False

    # Hard reject obvious non-name lines.
    non_name = {"BANK", "TOTAL", "GRAND", "PARTNER", "SIGNATURE", "PAGE"}
    if all(tok in non_name or tok in IGNORE_WORDS or tok in BAN_TOKENS for tok in tokens):
        return False

    cleaned = [t for t in tokens if t not in IGNORE_WORDS and t not in BAN_TOKENS]
    return len(cleaned) >= 1


def clean_wage_name(raw: str) -> str:
    raw_tokens = re.findall(r"[A-Z]+", raw.upper())
    if not raw_tokens:
        return ""

    while raw_tokens and raw_tokens[0] in BANK_PREFIXES:
        raw_tokens.pop(0)

    filtered = []
    for tok in raw_tokens:
        if tok in BAN_TOKENS or tok in IGNORE_WORDS:
            continue
        # Keep initials only when they are part of a real name.
        if len(tok) == 1 and len(raw_tokens) == 1:
            continue
        filtered.append(tok)

    if not filtered:
        return ""

    # Remove trailing single-letter fragments unless the whole name is just
    # initial + surname style.
    if len(filtered) >= 3:
        while filtered and len(filtered[-1]) == 1:
            filtered.pop()
    if not filtered:
        return ""
    return " ".join(filtered)


def extract_bank_name(narration: str) -> str:
    m = re.search(r"NEFT-([^-]+)", narration.upper())
    return m.group(1) if m else narration


def name_similarity(wage_name: str, bank_text: str) -> float:
    wage_tokens = normalize_name(wage_name).split()
    bank_tokens = normalize_name(extract_bank_name(bank_text)).split()
    generic = {
        "BANK", "STATE", "OF", "INDIA", "LTD", "LIMITED", "LLP", "PVT",
        "INDUSIND", "PUNJAB", "NATIONAL", "CANARA", "AXIS", "KOTAK",
        "IDFC", "HDFC", "ICICI", "SBIN", "CNRB", "IOBA", "UTIB",
        "INDB", "IPOS", "IOB", "SBI", "YES", "UCO", "FEDERAL",
    }
    filtered_wage = [t for t in wage_tokens if t not in generic] or wage_tokens
    filtered_bank = [t for t in bank_tokens if t not in generic] or bank_tokens
    if not filtered_bank or not filtered_wage:
        return 0.0
    matched = 0
    for bt in filtered_bank:
        best = 0.0
        for wt in filtered_wage:
            best = max(best, difflib.SequenceMatcher(None, bt, wt).ratio())
        if best > 0.85:
            matched += 1
    if matched == 0:
        return 0.0
    total = len(filtered_wage) + len(filtered_bank)
    base = 2.0 * matched / total
    bonus = 0.0
    if filtered_bank:
        first_b = filtered_bank[0]
        if any(difflib.SequenceMatcher(None, first_b, wt).ratio() > 0.85 for wt in filtered_wage):
            bonus += 0.1
    if matched >= 2:
        bonus += 0.1
    return min(base + bonus, 1.0)


def extract_wage_entries(pdf: Path) -> List[dict]:
    """Extract wage employees from multiple register layouts.

    Supports both layouts where the serial number appears before the net payable
    and layouts where the serial number appears after gross/deduction columns.
    """
    entries: List[dict] = []
    expected_serial = 1
    dec_amount_pattern = re.compile(r"\d[\d,]*\.\d{2}")
    int_amount_pattern = re.compile(r"\d[\d,]*")

    def parse_float(raw: str) -> Optional[float]:
        try:
            return float(raw.replace(',', ''))
        except Exception:
            return None

    def is_serial_line(line: str, expected: int) -> bool:
        return line.strip() == str(expected)

    def candidate_amounts_from_line(line: str) -> List[tuple[float, str]]:
        out = []
        for raw in dec_amount_pattern.findall(line):
            val = parse_float(raw)
            if val is not None:
                out.append((val, raw))
        if out:
            return out
        for raw in int_amount_pattern.findall(line):
            if not raw or raw == '0':
                continue
            val = parse_float(raw)
            if val is None:
                continue
            out.append((val, raw))
        return out

    def plausible_amount(val: float) -> bool:
        return 1.0 <= val <= 1000000.0

    def choose_net_amount(lines: List[str], serial_idx: int) -> Optional[tuple[float, str]]:
        # Prefer decimal values after the serial line. In many registers the net
        # payable appears immediately after serial / LWF columns.
        forward_hits: List[tuple[int, float, str]] = []
        for k in range(serial_idx + 1, min(len(lines), serial_idx + 8)):
            line = lines[k].strip()
            if not line or line.upper() in {'WD', 'WO', 'PD', 'PL', 'PH', 'CL', 'TOT', 'TOTAL'}:
                continue
            for val, raw in candidate_amounts_from_line(line):
                if '.' in raw and plausible_amount(val):
                    forward_hits.append((k, val, raw))
        if forward_hits:
            # take the first reasonable decimal after the serial
            _, val, raw = forward_hits[0]
            return val, raw

        # Fallback: use reasonable numeric values shortly before the serial.
        back_hits: List[tuple[int, float, str]] = []
        for k in range(max(0, serial_idx - 6), serial_idx):
            line = lines[k].strip()
            if not line or line.upper() in {'WD', 'WO', 'PD', 'PL', 'PH', 'CL', 'TOT', 'TOTAL'}:
                continue
            for val, raw in candidate_amounts_from_line(line):
                # ignore attendance/day counts and tiny codes
                if val < 1000 or not plausible_amount(val):
                    continue
                back_hits.append((k, val, raw))
        if back_hits:
            # Usually the first large figure before serial is the net payable.
            # If multiple values exist, prefer the earliest large value in the final block.
            _, val, raw = back_hits[0]
            return val, raw
        return None

    with fitz.open(str(pdf)) as doc:
        for page_no, page in enumerate(doc):
            lines = [ln.strip() for ln in page.get_text().split("\n") if ln.strip()]
            i = 0
            while i < len(lines):
                if not is_serial_line(lines[i], expected_serial):
                    i += 1
                    continue

                prev_start = max(0, i - 25)
                prev = lines[prev_start:i]
                wd_positions = [idx for idx, val in enumerate(prev) if val.upper() == 'WD']
                wd_idx = prev_start + wd_positions[-1] if wd_positions else i

                name_parts: List[str] = []
                j = wd_idx - 1
                lower_bound = max(0, wd_idx - 8)
                while j >= lower_bound:
                    cand = lines[j].strip()
                    if is_probable_name_line(cand):
                        name_parts.append(cand)
                        j -= 1
                        continue
                    if name_parts:
                        break
                    j -= 1

                # Fallback for layouts where WD is missing or OCR/text order is odd:
                # use the nearest probable text line before the serial.
                if not name_parts:
                    for j in range(i - 1, max(-1, i - 8), -1):
                        cand = lines[j].strip()
                        if is_probable_name_line(cand):
                            name_parts = [cand]
                            break

                cleaned = clean_wage_name(" ".join(reversed(name_parts)))
                net_info = choose_net_amount(lines, i)

                if cleaned and net_info:
                    amt, raw_amt = net_info
                    entries.append({
                        'id': len(entries),
                        'serial': expected_serial,
                        'page': page_no + 1,
                        'name': cleaned,
                        'amount': amt,
                        'amount_text': raw_amt,
                        'status': 'PENDING',
                        'selected_candidates': [],
                    })

                # Always advance when the expected serial is found so one bad row
                # cannot block every later employee.
                expected_serial += 1
                i += 1

    return entries

def group_lines(page: fitz.Page) -> List[Tuple[int, List, str]]:
    words = page.get_text("words")
    lines_dict = {}
    for w in words:
        ykey = int(round(w[1] / 3.0))
        lines_dict.setdefault(ykey, []).append(w)
    result = []
    for key, line_words in sorted(lines_dict.items()):
        line_words.sort(key=lambda x: x[0])
        text = " ".join(w[4] for w in line_words)
        result.append((key, line_words, text))
    return result


DATE_LINE_RE = re.compile(r"^\d{2}[-/]\d{2}[-/]\d{4}\b")


def extract_context(lines, idx, radius=2):
    return [lines[j][2] for j in range(max(0, idx - radius), min(len(lines), idx + radius + 1))]


def line_starts_transaction(text: str) -> bool:
    return bool(DATE_LINE_RE.match(text.strip()))


def transaction_range_for_line(lines: List[Tuple[int, List, str]], idx: int) -> Tuple[int, int]:
    start = idx
    while start > 0 and not line_starts_transaction(lines[start][2].strip()):
        start -= 1
    if not line_starts_transaction(lines[start][2].strip()):
        start = idx

    end = idx
    j = max(idx + 1, start + 1)
    while j < len(lines):
        nxt = lines[j][2].strip()
        if line_starts_transaction(nxt):
            break
        end = j
        j += 1
    return start, end


def build_transaction_blocks(lines: List[Tuple[int, List, str]]) -> List[dict]:
    blocks = []
    i = 0
    while i < len(lines):
        text = lines[i][2].strip()
        if not line_starts_transaction(text):
            i += 1
            continue
        start = i
        block_lines = [text]
        j = i + 1
        while j < len(lines):
            nxt = lines[j][2].strip()
            if line_starts_transaction(nxt):
                break
            block_lines.append(nxt)
            j += 1
        joined = " ".join(block_lines)
        amounts = re.findall(r"\d[\d,]*\.\d{2}", joined)
        blocks.append({
            "start_line": start,
            "end_line": j - 1,
            "lines": block_lines,
            "text": joined,
            "amounts": amounts,
        })
        i = j
    return blocks




def parse_amount_strings(amount_texts: List[str]) -> List[float]:
    values = []
    for n in amount_texts or []:
        try:
            values.append(float(str(n).replace(',', '')))
        except Exception:
            pass
    return values


def candidate_amount_distance(wage_amount: float, candidate: dict) -> float:
    values = [v for v in parse_amount_strings(candidate.get("amounts", [])) if 0 < v < max(wage_amount * 3, wage_amount + 10000)]
    if not values:
        return 10**9
    return min(abs(v - wage_amount) for v in values)


def choose_default_candidates(entry: dict, candidates: List[dict]) -> List[str]:
    if not candidates:
        return []
    wage_amount = float(entry["amount"])
    enriched = []
    for c in candidates:
        dist = candidate_amount_distance(wage_amount, c)
        enriched.append((c, dist))

    # Prefer the best score, then the closest amount.
    enriched.sort(key=lambda item: (-item[0]["score"], item[1]))
    top = enriched[0][0]
    top_dist = enriched[0][1]
    second_score = enriched[1][0]["score"] if len(enriched) > 1 else 0.0

    # 1) Strong obvious top match: pre-tick it even if amount parsing is slightly noisy.
    if top["score"] >= 0.995:
        return [str(top["candidate_id"])]

    # 2) Exact / near-exact single match should already be ticked for the user.
    exact_top = [str(c["candidate_id"]) for c, dist in enriched if c["score"] >= 0.98 and dist <= 2.0]
    if len(exact_top) == 1:
        return exact_top

    # 3) Very strong single-candidate match.
    if top["score"] >= 0.96 and top_dist <= 10.0 and (len(enriched) == 1 or top["score"] - second_score >= 0.01):
        return [str(top["candidate_id"])]

    # 4) Exact / near-exact single match among very strong candidates.
    exactish = [str(c["candidate_id"]) for c, dist in enriched if c["score"] >= 0.92 and dist <= 35]
    if len(exactish) == 1:
        return exactish

    # 5) Try a 2-line combo for split payments.
    shortlist = [(c, dist) for c, dist in enriched[:12] if c["score"] >= 0.70]
    parsed = []
    for c, dist in shortlist:
        vals = [v for v in parse_amount_strings(c.get("amounts", [])) if 0 < v < wage_amount + 2000]
        if not vals:
            continue
        best_val = min(vals, key=lambda v: abs(v - wage_amount))
        parsed.append((c, best_val, dist))

    best_combo = None
    for i in range(len(parsed)):
        c1, v1, _ = parsed[i]
        for j in range(i + 1, len(parsed)):
            c2, v2, _ = parsed[j]
            diff = abs((v1 + v2) - wage_amount)
            avg_score = (c1["score"] + c2["score"]) / 2
            if diff <= 25 and avg_score >= 0.82:
                if best_combo is None or (diff, -avg_score) < (best_combo[0], -best_combo[1]):
                    best_combo = (diff, avg_score, [str(c1["candidate_id"]), str(c2["candidate_id"])])
    if best_combo:
        return best_combo[2]

    # 6) Workable fallback: if the first candidate is clearly strongest, pre-tick it.
    if top["score"] >= 0.90 and top_dist <= 150 and (len(enriched) == 1 or top["score"] - second_score >= 0.03):
        return [str(top["candidate_id"])]

    return []



def should_auto_approve(entry: dict) -> bool:
    selected = entry.get("preselected_candidates", []) or []
    candidates = entry.get("candidates", []) or []
    if not selected or not candidates:
        return False

    by_id = {c["candidate_id"]: c for c in candidates}
    wage_amount = float(entry["amount"])

    # Single-candidate auto-approve only when it is exact/near-exact and clearly best.
    if len(selected) == 1:
        c = by_id.get(selected[0])
        if not c:
            return False
        dist = candidate_amount_distance(wage_amount, c)
        second_score = max([cand["score"] for cand in candidates if cand["candidate_id"] != c["candidate_id"]], default=0.0)
        return c["score"] >= 0.995 and dist <= 1.0 and (len(candidates) == 1 or c["score"] - second_score >= 0.05)

    # Two-candidate split payment auto-approve only when total is exact and both are strong.
    if len(selected) == 2:
        chosen = [by_id.get(cid) for cid in selected]
        if any(c is None for c in chosen):
            return False
        best_vals = []
        for c in chosen:
            vals = [v for v in parse_amount_strings(c.get("amounts", [])) if 0 < v < wage_amount + 2000]
            if not vals:
                return False
            best_vals.append(min(vals, key=lambda v: abs(v - wage_amount)))
        total = sum(best_vals)
        avg_score = sum(c["score"] for c in chosen) / 2
        return abs(total - wage_amount) <= 1.0 and avg_score >= 0.93

    return False

def analyze_bank_lines(bank_pdf: Path) -> List[dict]:
    data = []
    with fitz.open(str(bank_pdf)) as doc:
        for pidx in range(len(doc)):
            lines = group_lines(doc[pidx])
            for idx, (ykey, words, text) in enumerate(lines):
                amounts = re.findall(r"\d[\d,]*\.\d{2}", text)
                data.append({
                    "candidate_id": f"{pidx}:{idx}",
                    "page": pidx,
                    "line_index": idx,
                    "ykey": ykey,
                    "text": text,
                    "amounts": amounts,
                    "context": extract_context(lines, idx, radius=2),
                })
    return data



def score_candidate(wage_entry: dict, bank_line: dict) -> float:
    score = name_similarity(wage_entry["name"], bank_line["text"])
    amt = wage_entry["amount"]
    amount_bonus = 0.0
    combo_bonus = 0.0
    all_numbers = []
    for line in bank_line["context"]:
        all_numbers.extend(re.findall(r"\d[\d,]*\.\d{2}", line))
    values = []
    for n in all_numbers:
        try:
            values.append(float(n.replace(',', '')))
        except Exception:
            pass
    if values:
        best_diff = min(abs(v - amt) for v in values)
        if best_diff <= 1:
            amount_bonus = 0.35
        elif best_diff <= 250:
            amount_bonus = 0.22
        elif best_diff <= 1000:
            amount_bonus = 0.12
        small = [v for v in values if 0 < v < amt + 1000]
        for i, a in enumerate(small):
            for b in small[i + 1:]:
                if abs((a + b) - amt) <= 250:
                    combo_bonus = max(combo_bonus, 0.20)
    return min(score + amount_bonus + combo_bonus, 1.0)



def build_review_queue(wage_entries: List[dict], bank_lines: List[dict]) -> List[dict]:
    queue = []
    for entry in wage_entries:
        candidates = []
        for line in bank_lines:
            score = score_candidate(entry, line)
            if score < CONFIG["min_match_score"]:
                continue
            candidates.append({
                "candidate_id": line["candidate_id"],
                "page": line["page"] + 1,
                "line_index": line["line_index"],
                "text": line["text"],
                "context": line["context"],
                "score": round(score, 3),
                "amounts": line["amounts"],
            })
        candidates.sort(key=lambda x: x["score"], reverse=True)
        entry_copy = dict(entry)
        entry_copy["candidates"] = candidates[: CONFIG["review_candidates"]]
        entry_copy["preselected_candidates"] = choose_default_candidates(entry_copy, entry_copy["candidates"])
        queue.append(entry_copy)
    return queue


def state_path(run_id: str) -> Path:
    return STATE_DIR / f"{run_id}.json"


def save_state(data: dict) -> None:
    state_path(data["run_id"]).write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_state(run_id: str) -> Optional[dict]:
    path = state_path(run_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None




def get_active_state() -> Optional[dict]:
    run_id = session.get("run_id")
    if not run_id:
        return None
    state = load_state(run_id)
    if not state:
        session.pop("run_id", None)
        return None
    return state

def current_entry_index(state: dict) -> Optional[int]:
    for idx, entry in enumerate(state["review_queue"]):
        if entry["status"] == "PENDING":
            return idx
    return None


def generate_outputs(state: dict) -> None:
    bank_pdf = Path(state["bank_pdf"])
    out_pdf = Path(state["output_pdf"])
    out_csv = Path(state["output_csv"])
    selected_ids = []
    for e in state["review_queue"]:
        selected_ids.extend(e.get("selected_candidates", []))
    selected_map = {cid: True for cid in selected_ids}

    with fitz.open(str(bank_pdf)) as doc:
        for candidate_id in selected_map:
            try:
                pidx, line_index = map(int, candidate_id.split(":"))
            except ValueError:
                continue
            if pidx < 0 or pidx >= len(doc):
                continue
            page = doc[pidx]
            lines = group_lines(page)
            if line_index < 0 or line_index >= len(lines):
                continue

            start_line, end_line = transaction_range_for_line(lines, line_index)
            word_rects = []
            for line_no in range(start_line, end_line + 1):
                line_words = lines[line_no][1]
                if not line_words:
                    continue
                for w in line_words:
                    word_rects.append(fitz.Rect(w[0], w[1], w[2], w[3]))

            if word_rects:
                annot = page.add_highlight_annot(word_rects)
                annot.set_colors(stroke=(1.0, 1.0, 0.0))
                annot.set_opacity(0.95)
                annot.update()
        doc.save(str(out_pdf), garbage=4, deflate=True, clean=True)

    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Name", "Wage Amount", "Status", "Selected Candidates", "Selected Scores"])
        for entry in state["review_queue"]:
            selected_texts = []
            selected_scores = []
            for cid in entry.get("selected_candidates", []):
                for c in entry.get("candidates", []):
                    if c["candidate_id"] == cid:
                        selected_texts.append(c["text"])
                        selected_scores.append(str(c["score"]))
                        break
            writer.writerow([
                entry["name"],
                entry["amount_text"],
                entry["status"],
                " | ".join(selected_texts),
                " | ".join(selected_scores),
            ])



@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        wage_file = request.files.get("wage_pdf")
        bank_file = request.files.get("bank_pdf")
        if not wage_file or not bank_file:
            flash("Please upload both PDFs.", "error")
            return redirect(url_for("index"))

        run_id = uuid.uuid4().hex[:12]
        wage_path = UPLOAD_DIR / f"{run_id}_{secure_filename(wage_file.filename or 'wages.pdf')}"
        bank_path = UPLOAD_DIR / f"{run_id}_{secure_filename(bank_file.filename or 'bank.pdf')}"
        wage_file.save(wage_path)
        bank_file.save(bank_path)

        wage_entries = extract_wage_entries(wage_path)
        bank_lines = analyze_bank_lines(bank_path)
        review_queue = build_review_queue(wage_entries, bank_lines)
        if not review_queue:
            flash("No employees could be extracted from the wage PDF. Please use the updated parser build.", "error")
            return redirect(url_for("index"))

        state = {
            "run_id": run_id,
            "wage_pdf": str(wage_path),
            "bank_pdf": str(bank_path),
            "output_pdf": str(OUTPUT_DIR / f"highlighted_{run_id}.pdf"),
            "output_csv": str(OUTPUT_DIR / f"review_report_{run_id}.csv"),
            "review_queue": review_queue,
        }
        save_state(state)
        session["run_id"] = run_id
        return redirect(url_for("review"))
    return render_template("index.html")


@app.route("/review")
def review():
    state = get_active_state()
    if not state:
        flash("Session expired or files were removed. Please upload the PDFs again.", "error")
        return redirect(url_for("index"))

    # Recompute default selections on every review load so strong matches are visibly pre-ticked
    # even if an older state file was created before the latest auto-select logic.
    pending_changed = False
    for _entry in state.get("review_queue", []):
        if _entry.get("status") == "PENDING" and _entry.get("candidates"):
            defaults = choose_default_candidates(_entry, _entry.get("candidates", []))
            if defaults != (_entry.get("preselected_candidates") or []):
                _entry["preselected_candidates"] = defaults
                pending_changed = True

    # Do not silently auto-approve on page load.
    # Strong matches are pre-selected for the user, but review stays sequential
    # so the counter moves 1 by 1 and nothing is skipped unexpectedly.
    if pending_changed:
        save_state(state)

    idx = current_entry_index(state)
    total = len(state["review_queue"])
    done = len([e for e in state["review_queue"] if e["status"] != "PENDING"])
    if idx is None:
        if total == 0:
            flash("No employees were loaded for review. Please re-upload the wage and statement PDFs.", "error")
            return redirect(url_for("index"))
        generate_outputs(state)
        yes_count = len([e for e in state["review_queue"] if e["status"] == "APPROVED"])
        skip_count = len([e for e in state["review_queue"] if e["status"] == "SKIPPED"])
        return render_template("done.html", total=total, yes_count=yes_count, skip_count=skip_count)
    entry = state["review_queue"][idx]
    return render_template("review.html", entry=entry, idx=idx + 1, total=total, done=done)


@app.route("/decide", methods=["POST"])
def decide():
    state = get_active_state()
    if not state:
        flash("Session expired or files were removed. Please upload the PDFs again.", "error")
        return redirect(url_for("index"))
    idx = current_entry_index(state)
    if idx is None:
        return redirect(url_for("review"))
    action = request.form.get("action")
    entry = state["review_queue"][idx]
    if action == "approve":
        candidate_ids = request.form.getlist("candidate_id")
        if not candidate_ids:
            flash("Select at least one candidate first.", "error")
            return redirect(url_for("review"))
        entry["status"] = "APPROVED"
        entry["selected_candidates"] = candidate_ids
    elif action == "skip":
        entry["status"] = "SKIPPED"
        entry["selected_candidates"] = []
    else:
        flash("Unknown action.", "error")
    save_state(state)
    return redirect(url_for("review"))


@app.route("/download/pdf")
def download_pdf():
    state = get_active_state()
    if not state:
        flash("Session expired or files were removed. Please upload the PDFs again.", "error")
        return redirect(url_for("index"))
    output = Path(state["output_pdf"])
    if not output.exists():
        generate_outputs(state)
    return send_file(output, as_attachment=True)


@app.route("/download/csv")
def download_csv():
    state = get_active_state()
    if not state:
        flash("Session expired or files were removed. Please upload the PDFs again.", "error")
        return redirect(url_for("index"))
    output = Path(state["output_csv"])
    try:
        if not output.exists():
            generate_outputs(state)
    except Exception:
        flash("Could not generate the review CSV for this run. Please upload the PDFs again.", "error")
        return redirect(url_for("index"))
    if not output.exists():
        flash("Review CSV was not created. Please upload the PDFs again.", "error")
        return redirect(url_for("index"))
    return send_file(output, as_attachment=True)


@app.route("/download/pages-pdf")
def download_pages_pdf():
    """Download a slim PDF: page 1 always + only pages that contain highlights."""
    state = get_active_state()
    if not state:
        flash("Session expired or files were removed. Please upload the PDFs again.", "error")
        return redirect(url_for("index"))

    # Make sure the full highlighted PDF exists first
    full_output = Path(state["output_pdf"])
    if not full_output.exists():
        generate_outputs(state)

    # Collect the (0-based) page indices that were highlighted
    highlighted_pages: set[int] = set()
    for e in state["review_queue"]:
        for cid in e.get("selected_candidates", []):
            try:
                pidx, _ = map(int, cid.split(":"))
                highlighted_pages.add(pidx)
            except ValueError:
                continue

    # Always include page 0 (first page of bank statement)
    highlighted_pages.add(0)
    pages_to_keep = sorted(highlighted_pages)

    # Build the slim PDF from the already-highlighted output
    slim_path = Path(state["output_pdf"]).parent / f"pages_only_{state["run_id"]}.pdf"
    with fitz.open(str(full_output)) as src:
        out_doc = fitz.open()
        for pidx in pages_to_keep:
            if pidx < len(src):
                out_doc.insert_pdf(src, from_page=pidx, to_page=pidx)
        out_doc.save(str(slim_path), garbage=4, deflate=True, clean=True)

    return send_file(slim_path, as_attachment=True,
                     download_name="highlighted_pages_only.pdf")


if __name__ == "__main__":
    app.run(debug=True)
