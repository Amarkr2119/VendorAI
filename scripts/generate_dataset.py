#!/usr/bin/env python3
"""Generate the VendorFlow AI synthetic dataset (deterministic, seed-controlled).

Outputs (data/synthetic/):
  applications.json          submitted data + document references, one entry per application
  ground_truth.json          the true value of every extractable field as PRINTED on each document
  manifest.csv               scenario, expected completeness result, expected decisions, reason codes, actions
  simulation_fixtures.json   what the SIMULATED providers will answer (Phase 4). Not real verification data.
  internal_blocklist.json    fictional internal blocklist used by the documented rejection scenario (US-010)
  documents/<ref>/*.pdf|png  registration form, certificates, bank proof

Everything is fictional. Company names, people, banks and identifiers are invented; identifiers are
format-valid (checksums included) so format checks pass or fail ONLY where a scenario says so.
Every document is footed "SYNTHETIC SAMPLE - NOT A REAL DOCUMENT".
"""
from __future__ import annotations

import csv
import io
import json
import random
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np                                   # noqa: E402
from PIL import Image, ImageEnhance, ImageFilter      # noqa: E402
from reportlab.lib.pagesizes import A4               # noqa: E402
from reportlab.pdfgen import canvas                  # noqa: E402

from vendorflow.validation import identifiers as ids  # noqa: E402

OUT = ROOT / "data" / "synthetic"
SEED = 20261110
FOOTER = "SYNTHETIC SAMPLE - NOT A REAL DOCUMENT"

FORM_LABELS = {
    "legal_name": "Legal Name", "trade_name": "Trade Name", "entity_type": "Entity Type", "address": "Registered Address",
    "contact_email": "Contact Email", "bank_name": "Bank Name", "bank_holder_name": "Account Holder Name",
    "bank_account_number": "Account Number", "pan": "PAN", "gstin": "GSTIN", "cin": "CIN", "ifsc": "IFSC Code",
    "ein": "EIN", "state_of_registration": "State of Registration", "state_file_number": "State File Number",
    "routing_number": "ABA Routing Number", "company_number": "Company Number", "vat_number": "VAT Number",
    "sort_code": "Sort Code",
}
FORM_ORDER = {
    "IN": ["legal_name", "trade_name", "entity_type", "address", "contact_email", "pan", "gstin", "cin",
           "bank_name", "bank_holder_name", "bank_account_number", "ifsc"],
    "US": ["legal_name", "trade_name", "entity_type", "address", "contact_email", "ein", "state_of_registration",
           "state_file_number", "bank_name", "bank_holder_name", "bank_account_number", "routing_number"],
    "GB": ["legal_name", "trade_name", "entity_type", "address", "contact_email", "company_number", "vat_number",
           "bank_name", "bank_holder_name", "bank_account_number", "sort_code"],
}
COUNTRY_LABEL = {"IN": "India", "US": "USA", "GB": "United Kingdom"}


# ============================================================================= identifier generation
class Gen:
    def __init__(self, seed: int):
        self.r = random.Random(seed)
        self.used: set = set()

    def _unique(self, make):
        while True:
            v = make()
            if v not in self.used:
                self.used.add(v)
                return v

    def digits(self, n: int) -> str:
        return "".join(self.r.choice("0123456789") for _ in range(n))

    def letters(self, n: int) -> str:
        return "".join(self.r.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(n))

    def pan(self, name: str) -> str:
        return self._unique(lambda: self.letters(3) + "C" + name[0].upper() + self.digits(4) + self.letters(1))

    def gstin(self, state: str, pan: str, entity: str = "1") -> str:
        first14 = f"{state}{pan}{entity}Z"
        return first14 + ids.gstin_check_char(first14)

    def cin(self) -> str:
        return self._unique(lambda: f"U{self.digits(5)}{self.r.choice(['MH', 'KA', 'DL', 'GJ', 'TN'])}"
                                    f"{self.r.choice(['2012', '2015', '2018', '2020'])}PTC{self.digits(6)}")

    def ifsc(self) -> str:
        return "SYNB0" + self.digits(6)                      # SYNB = fictional "Synthetic National Bank"

    def ein(self) -> str:
        return self._unique(lambda: f"{self.r.choice(['12', '31', '45', '52', '61', '83'])}-{self.digits(7)}")

    def routing(self) -> str:
        first8 = "9" + self.digits(7)                         # leading 9 -> not a real Federal Reserve routing range
        return first8 + ids.aba_check_digit(first8)

    def uk_vat(self, new_style: bool = False) -> str:
        f7 = self.digits(7)
        return f7 + ids.uk_vat_check_digits(f7, new_style)

    def sort_code(self) -> str:
        return f"98-{self.digits(2)}-{self.digits(2)}"        # 98-xx-xx: not an allocated bank range

    def email(self, name: str) -> str:
        import re
        words = [w for w in re.findall(r"[a-z0-9]+", name.lower()) if w not in {"and"}]
        return f"accounts@{'-'.join(words[:2])}.example"


# ============================================================================= document rendering
@dataclass
class DocSpec:
    doc_type: str
    filename: str
    title: str
    subtitle: str
    lines: List[Tuple[str, str, Optional[str]]]        # (label, value, field-or-None)
    scan: str = "none"                                  # none | clean | poor | bad
    fmt: str = "pdf"                                    # pdf | png


def render_pdf(path: Path, title: str, subtitle: str, lines, doc_ref: str) -> None:
    c = canvas.Canvas(str(path), pagesize=A4)
    w, h = A4
    c.setTitle(title)
    c.setFont("Helvetica-Bold", 15)
    c.drawString(50, h - 70, title)
    c.setFont("Helvetica", 9)
    c.drawString(50, h - 88, subtitle)
    c.line(50, h - 96, w - 50, h - 96)
    y = h - 128
    c.setFont("Helvetica", 11)
    for label, value, _ in lines:
        c.drawString(50, y, f"{label}: {value}")
        y -= 22
    c.line(50, 70, w - 50, 70)
    c.setFont("Helvetica", 8)
    c.drawString(50, 56, f"{FOOTER}   |   Doc Ref: {doc_ref}")
    c.save()


def pdf_to_image(path: Path, dpi: int) -> Image.Image:
    import pypdfium2 as pdfium
    pdf = pdfium.PdfDocument(str(path))
    img = pdf[0].render(scale=dpi / 72).to_pil().convert("L")
    pdf.close()
    return img


# (scale, rotation_deg, blur_radius, contrast, noise_sigma, jpeg_quality)
SCAN_LEVELS = {
    "clean": (1.00, 0.4, 0.0, 1.00, 4.0, 85),     # good office scanner: OCR is reliable
    "poor": (0.60, 1.5, 1.0, 0.78, 16.0, 40),     # tuned empirically: mostly readable, identifiers sometimes misread
    "bad": (0.55, 1.8, 1.25, 0.72, 22.0, 28),     # effectively unreadable (used by tests of the unreadable path)
}


def degrade(img: Image.Image, level: str, seed: int) -> Image.Image:
    """Simulate a scan at a named quality level (see SCAN_LEVELS)."""
    scale, rot, blur, contrast, sigma, jpeg = SCAN_LEVELS[level]
    rng = np.random.default_rng(seed)
    if scale != 1.0:
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.BILINEAR)
    img = img.rotate(rot, resample=Image.BICUBIC, fillcolor=255)
    if blur:
        img = img.filter(ImageFilter.GaussianBlur(blur))
    if contrast != 1.0:
        img = ImageEnhance.Contrast(img).enhance(contrast)
    arr = np.asarray(img, dtype=np.float32) + rng.normal(0, sigma, size=(img.size[1], img.size[0]))
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=jpeg)
    return Image.open(io.BytesIO(buf.getvalue())).convert("L")


def write_document(spec: DocSpec, folder: Path, ref: str, seed: int) -> Path:
    path = folder / spec.filename
    import zlib
    doc_ref = f"SYN-{ref}-{zlib.crc32(spec.doc_type.encode()) % 10000:04d}"   # neutral: must not leak the document type
    if spec.scan == "none":
        render_pdf(path, spec.title, spec.subtitle, spec.lines, doc_ref)
        return path
    tmp = folder / f"_tmp_{spec.filename}.pdf"
    render_pdf(tmp, spec.title, spec.subtitle, spec.lines, doc_ref)
    img = degrade(pdf_to_image(tmp, 200), spec.scan, seed)
    tmp.unlink()
    if spec.fmt == "png":
        img.save(path, "PNG")
    else:
        img.convert("RGB").save(path, "PDF", resolution=150.0)         # image-only PDF: no text layer
    return path


# ============================================================================= document content
def L(label: str, value: str, field_name: Optional[str] = None):
    return (label, value, field_name)


def doc_form(country: str, c: Dict[str, Any]) -> DocSpec:
    lines = [L("Country", COUNTRY_LABEL[country])]
    for f in FORM_ORDER[country]:
        v = c["form"].get(f, "")
        if v:
            lines.append(L(FORM_LABELS[f], v, f))
    if country == "IN":
        lines.append(L("GST Registered", "No" if c["declarations"].get("gst_not_registered") else "Yes"))
    if country == "GB":
        lines.append(L("VAT Registered", "No" if c["declarations"].get("vat_not_registered") else "Yes"))
    lines.append(L("Submission Date", "10/11/2026"))
    return DocSpec("registration_form", "registration_form.pdf", "VENDOR REGISTRATION FORM",
                   "Supplier onboarding - applicant declaration", lines)


def doc_reg_cert(country: str, c: Dict[str, Any]) -> DocSpec:
    n = c["cert_name"]
    if country == "IN":
        return DocSpec("registration_certificate", "registration_certificate.pdf", "CERTIFICATE OF INCORPORATION",
                       "Ministry of Corporate Affairs (synthetic issuer)",
                       [L("Corporate Identity Number", c["cin"], "cin"), L("Company Name", n, "legal_name"),
                        L("Registered Office", c["address"], "address"), L("Date of Incorporation", "12/03/2018")])
    if country == "US":
        return DocSpec("registration_certificate", "registration_certificate.pdf", "ARTICLES OF ORGANIZATION",
                       f"State of {c['state']} - Secretary of State (synthetic issuer)",
                       [L("Entity Name", n, "legal_name"), L("State File Number", c["state_file_number"], "state_file_number"),
                        L("State of Formation", c["state"], "state_of_registration"),
                        L("Principal Address", c["address"], "address"), L("Date Filed", "03/09/2019")])
    return DocSpec("registration_certificate", "registration_certificate.pdf",
                   "CERTIFICATE OF INCORPORATION OF A PRIVATE LIMITED COMPANY", "Companies House (synthetic issuer)",
                   [L("Company Number", c["company_number"], "company_number"), L("Company Name", n, "legal_name"),
                    L("Registered Office", c["address"], "address"), L("Date of Incorporation", "21/05/2016")])


def doc_tax_cert(country: str, c: Dict[str, Any]) -> DocSpec:
    n = c["cert_name"]
    if country == "IN":
        return DocSpec("tax_certificate", "tax_certificate.pdf", "GST REGISTRATION CERTIFICATE",
                       "Form GST REG-06 (synthetic issuer)",
                       [L("Registration Number", c["gst_cert_gstin"], "gstin"), L("Legal Name", n, "legal_name"),
                        L("Trade Name", c["cert_trade_name"], "trade_name"),
                        L("Address of Principal Place of Business", c["address"], "address"),
                        L("Date of Liability", "01/07/2017"), L("Type of Registration", "Regular")])
    if country == "US":
        return DocSpec("tax_certificate", "tax_certificate.pdf", "FORM W-9",
                       "Request for Taxpayer Identification Number and Certification (synthetic sample)",
                       [L("Name (as shown on your income tax return)", n, "legal_name"),
                        L("Business name/disregarded entity name", c["cert_trade_name"], "trade_name"),
                        L("Federal tax classification", c["entity_type"], "entity_type"),
                        L("Address", c["address"], "address"),
                        L("Employer identification number (EIN)", c["ein"], "ein")])
    return DocSpec("tax_certificate", "tax_certificate.pdf", "VAT REGISTRATION CERTIFICATE",
                   "HM Revenue and Customs (synthetic issuer)",
                   [L("VAT Registration Number", c["vat_display"], "vat_number"), L("Legal Name", n, "legal_name"),
                    L("Address", c["address"], "address"), L("Effective Date of Registration", "01/04/2019")])


def doc_pan_proof(c: Dict[str, Any]) -> DocSpec:
    return DocSpec("tax_id_proof", "tax_id_proof.pdf", "INCOME TAX DEPARTMENT - PERMANENT ACCOUNT NUMBER CARD",
                   "Government of India (synthetic issuer)",
                   [L("Name", c["cert_name"], "legal_name"), L("Permanent Account Number", c["pan"], "pan"),
                    L("Date of Incorporation", "12/03/2018")])


def doc_bank_proof(country: str, c: Dict[str, Any]) -> DocSpec:
    lines = [L("Bank Name", c["bank_name"], "bank_name"), L("Branch", "Central Branch"),
             L("Account Holder Name", c["bank_proof_holder"], "bank_holder_name"),
             L("Account Number", c["bank_account_number"], "bank_account_number")]
    if country == "IN":
        lines.append(L("IFSC Code", c["ifsc"], "ifsc"))
    if country == "US":
        lines.append(L("ABA Routing Number", c["routing_number"], "routing_number"))
    if country == "GB":
        lines.append(L("Sort Code", c["sort_code"], "sort_code"))
    lines.append(L("Account Type", "Current / Business"))
    return DocSpec("bank_proof", "bank_proof.pdf", "BANK ACCOUNT CONFIRMATION LETTER",
                   "Issued by the bank at the request of the account holder (synthetic)", lines)


# ============================================================================= company builders
ADDR_IN = ["Plot 14, Phase II, Industrial Area, Pune 411019, Maharashtra", "27 Lake View Road, Indiranagar, Bengaluru 560038, Karnataka",
           "B-9 Okhla Industrial Estate, New Delhi 110020", "12 Ring Road Crossing, Surat 395002, Gujarat",
           "88 Anna Salai, Teynampet, Chennai 600018, Tamil Nadu", "4th Floor, Harbor House, Fort, Mumbai 400001, Maharashtra",
           "31 Civil Lines, Nagpur 440001, Maharashtra", "Unit 6, Tech Park Road, Hyderabad 500081, Telangana",
           "101 Park Street, Kolkata 700016, West Bengal", "22 Sector 18, Gurugram 122015, Haryana"]
ADDR_US = ["1200 Larkspur Avenue, Denver, CO 80202", "845 Harbor Point Road, Portland, OR 97201", "300 Quarry Street, Austin, TX 78701",
           "77 Elm Grove Lane, Columbus, OH 43215", "19 Lakeshore Drive, Chicago, IL 60601", "510 Pine Ridge Court, Boise, ID 83702",
           "62 Mill Race Road, Raleigh, NC 27601", "930 Bayview Boulevard, Tampa, FL 33602", "4 Foundry Way, Pittsburgh, PA 15222",
           "2101 Mesa Verde Drive, Phoenix, AZ 85004"]
ADDR_GB = ["14 Millbrook Lane, Leeds LS1 4AB", "7 Quayside Row, Bristol BS1 5TT", "52 Canal Street, Manchester M1 3HE",
           "3 Harrow Bridge Road, Sheffield S1 2AA", "21 Ashdown Close, Brighton BN1 4GH", "9 Pennine Way, Huddersfield HD1 2QN",
           "40 Calder Mill, Halifax HX1 1AA", "18 Fernside Road, Norwich NR1 3DD", "6 Marsh Lane, Cambridge CB2 1AB",
           "11 Quillon Yard, Oxford OX1 2JD"]


def in_company(g: Gen, name: str, addr: str, state: str = "27") -> Dict[str, Any]:
    pan = g.pan(name)
    gstin = g.gstin(state, pan)
    acct = g.digits(12)
    return dict(country="IN", legal_name=name, cert_name=name, trade_name=name, cert_trade_name=name,
                entity_type="Private Limited Company", address=addr, contact_email=g.email(name), pan=pan, gstin=gstin,
                gst_cert_gstin=gstin, cin=g.cin(), ifsc=g.ifsc(), bank_name="Synthetic National Bank", bank_holder_name=name,
                bank_proof_holder=name, bank_account_number=acct, declarations={"gst_not_registered": False}, state=None)


def us_company(g: Gen, name: str, addr: str, state: str = "Colorado", entity: str = "LLC") -> Dict[str, Any]:
    return dict(country="US", legal_name=name, cert_name=name, trade_name=name, cert_trade_name=name, entity_type=entity,
                address=addr, contact_email=g.email(name), ein=g.ein(), state=state, state_file_number=g._unique(lambda: "2019" + g.digits(7)),
                routing_number=g.routing(), bank_name="Demo Heritage Bank", bank_holder_name=name, bank_proof_holder=name,
                bank_account_number=g.digits(10), declarations={})


def gb_company(g: Gen, name: str, addr: str, new_vat: bool = False) -> Dict[str, Any]:
    vat = g.uk_vat(new_vat)
    return dict(country="GB", legal_name=name, cert_name=name, trade_name=name, cert_trade_name=name,
                entity_type="Private Limited Company", address=addr, contact_email=g.email(name),
                company_number=g._unique(lambda: g.digits(8)), vat_number="GB" + vat, vat_display=f"GB {vat[:3]} {vat[3:7]} {vat[7:]}",
                sort_code=g.sort_code(), bank_name="Demo Britannia Bank", bank_holder_name=name, bank_proof_holder=name,
                bank_account_number=g.digits(8), declarations={"vat_not_registered": False})


FORM_KEYS = {f for fs in FORM_ORDER.values() for f in fs}


def finalize(c: Dict[str, Any]) -> Dict[str, Any]:
    """Derive the submitted form values from the company dict (after scenario overrides)."""
    form = {k: str(v) for k, v in c.items() if k in FORM_KEYS and v not in (None, "")}
    if c["country"] == "US":
        form["state_of_registration"] = c["state"]
    c["form"] = form
    return c


# ============================================================================= scenarios
@dataclass
class App:
    ref: str
    country: str
    scenario: str
    company: Dict[str, Any]
    drop_docs: List[str] = field(default_factory=list)
    scans: Dict[str, Tuple[str, str]] = field(default_factory=dict)       # doc_type -> (scan level, fmt)
    wrong_slot: Dict[str, str] = field(default_factory=dict)              # declared type -> content doc type
    expected: Dict[str, str] = field(default_factory=dict)


def E(**kw):
    base = dict(missing_fields="", missing_documents="", wrong_type_documents="", edge_case="", demo_priority="",
                auto_decision_demo="Approved", reason_codes="", final_after_review="", production_decision="Pending",
                required_actions="", notes="")
    base.update(kw)
    return base


def build_apps(g: Gen) -> List[App]:
    apps: List[App] = []
    # --------------------------------------------------------------------- INDIA
    c = in_company(g, "Kestrel Logistics Private Limited", ADDR_IN[0])
    in001 = c
    apps.append(App("IN-001", "IN", "Clean, complete application", c, expected=E(
        notes="Happy path. PRODUCTION policy needs LIVE verification, so it stays Pending there (VERIFICATION_NOT_LIVE).",
        reason_codes="", production_decision="Pending (VERIFICATION_NOT_LIVE)")))

    c = in_company(g, "Lumen Software Private Limited", ADDR_IN[1], "29")
    c["legal_name"] = "Lumen Software Pvt. Ltd."
    c["bank_holder_name"] = c["bank_proof_holder"] = "LUMEN SOFTWARE PRIVATE LIMITED"
    c["trade_name"] = "Lumen Software Pvt. Ltd."
    apps.append(App("IN-002", "IN", "Clean; name variants (Pvt. Ltd. vs Private Limited, upper case bank holder); clean scanned bank proof",
                    c, scans={"bank_proof": ("clean", "pdf")}, expected=E(
                        notes="Normalization must treat the variants as equivalent. Exercises OCR on a clean scan.")))

    c = in_company(g, "Banyan Foods Private Limited", ADDR_IN[2], "07")
    c["gstin"] = c["gst_cert_gstin"] = ""
    apps.append(App("IN-003", "IN", "Edge 1: GSTIN missing and no GST certificate", c, drop_docs=["tax_certificate"], expected=E(
        edge_case="1 Missing information", demo_priority="1", missing_fields="gstin", missing_documents="tax_certificate",
        auto_decision_demo="Pending", reason_codes="MISSING_FIELD;MISSING_DOCUMENT",
        required_actions="Provide GSTIN and GST certificate, or a declaration of non-registration")))

    c = in_company(g, "Orchid Packaging Private Limited", ADDR_IN[3], "24")
    apps.append(App("IN-004", "IN", "Edge 1: GST certificate not uploaded (GSTIN given)", c, drop_docs=["tax_certificate"], expected=E(
        edge_case="1 Missing information", demo_priority="1", missing_documents="tax_certificate", auto_decision_demo="Pending",
        reason_codes="MISSING_DOCUMENT", required_actions="Upload the GST registration certificate")))

    c = in_company(g, "Tarang Engineering Private Limited", ADDR_IN[4], "33")
    bad = c["gstin"][:-1] + ("A" if c["gstin"][-1] != "A" else "B")
    c["gstin"] = c["gst_cert_gstin"] = bad
    apps.append(App("IN-005", "IN", "Edge 3: GSTIN fails check character (form and documents consistently show it)", c, expected=E(
        edge_case="3 Invalid tax ID", demo_priority="2", auto_decision_demo="Pending", reason_codes="INVALID_FORMAT",
        required_actions="Correct the GSTIN (15 characters, valid state code, embedded PAN, valid check character)",
        notes="Documents faithfully show the invalid value, so extraction must NOT be flagged low-confidence; validation reports INVALID_FORMAT.")))

    c = in_company(g, "Vayu Cooling Private Limited", ADDR_IN[5], "27")
    other_pan = g.pan("Zeta Cooling")
    c["gstin"] = c["gst_cert_gstin"] = g.gstin("27", other_pan)
    apps.append(App("IN-006", "IN", "Edge 3: GSTIN valid but its embedded PAN differs from the submitted PAN", c, expected=E(
        edge_case="3 Invalid tax ID", auto_decision_demo="Pending", reason_codes="INVALID_FORMAT",
        required_actions="Confirm the PAN and GSTIN belong to the same entity (GSTIN characters 3-12 must equal the PAN)")))

    c = in_company(g, "Meghdoot Cables Private Limited", ADDR_IN[6], "27")
    c["bank_holder_name"] = c["bank_proof_holder"] = "Shree Ganesh Traders"
    apps.append(App("IN-007", "IN", "Edge 2: bank account holder is a different business", c, expected=E(
        edge_case="2 Bank/company name mismatch", demo_priority="3", auto_decision_demo="Pending", reason_codes="NAME_MISMATCH_BANK",
        required_actions="Explain the relationship to the account holder or supply an account in the company's name; routed for manual review")))

    c = in_company(g, "Kestrel Freight Services Private Limited", ADDR_IN[7], "27")
    c["pan"] = in001["pan"]
    c["gstin"] = c["gst_cert_gstin"] = in001["gstin"]
    apps.append(App("IN-008", "IN", "Edge 4: same GSTIN and PAN as IN-001 under a different company name", c, expected=E(
        edge_case="4 Duplicate vendor", demo_priority="4", auto_decision_demo="Pending", reason_codes="DUPLICATE_SUSPECTED",
        required_actions="Reviewer confirms whether this is the same vendor; existing application IN-001 is referenced",
        notes="Process IN-001 first.")))

    # Degraded scan with OCR-confusable characters in the identifiers
    for _ in range(400):
        pan = g.pan("Sagar Chemicals")
        gst = g.gstin("27", pan, "1")
        if sum(ch in "O0I1S5B8Z2G6" for ch in gst + pan) >= 8:
            break
    c = in_company(g, "Sagar Chemicals Private Limited", ADDR_IN[8], "27")
    c["pan"] = pan
    c["gstin"] = c["gst_cert_gstin"] = gst
    apps.append(App("IN-009", "IN", "Edge 6: poor-quality scans of the GST certificate and PAN proof", c,
                    scans={"tax_certificate": ("poor", "pdf"), "tax_id_proof": ("poor", "png")}, expected=E(
                        edge_case="6 Low-confidence extraction", demo_priority="", auto_decision_demo="Pending",
                        reason_codes="LOW_CONFIDENCE_EXTRACTION",
                        required_actions="Reviewer compares the extracted identifier with the document image, corrects it and reprocesses",
                        notes="OCR may misread O/0, I/1, S/5, B/8. Acceptable outcomes: correct value OR value flagged LOW. Never a confident wrong value.")))

    c = in_company(g, "Neelam Pharma Private Limited", ADDR_IN[9], "06")
    apps.append(App("IN-010", "IN", "Simulated provider reports the GSTIN as CANCELLED (must NOT cause Rejected)", c, expected=E(
        edge_case="Simulated negative", auto_decision_demo="Pending", reason_codes="VERIFICATION_SIMULATED_NEGATIVE",
        required_actions="Obtain evidence from the vendor / a LIVE check; simulated evidence can never establish disqualification",
        production_decision="Pending",
        notes="Policy: simulated evidence cannot establish a disqualifying condition. Pending in both DEMO and PRODUCTION.")))

    # --------------------------------------------------------------------- USA
    c = us_company(g, "Prairie Wind Analytics LLC", ADDR_US[0], "Colorado")
    us001 = c
    apps.append(App("US-001", "US", "Clean, complete application", c, expected=E(production_decision="Pending (VERIFICATION_NOT_LIVE)")))

    c = us_company(g, "Redwood Harbor Supply LLC", ADDR_US[1], "Oregon")
    c["legal_name"] = "Redwood Harbor Supply Limited Liability Company"
    c["trade_name"] = "Redwood Harbor Supply Limited Liability Company"
    c["bank_holder_name"] = c["bank_proof_holder"] = "REDWOOD HARBOR SUPPLY LLC"
    apps.append(App("US-002", "US", "Clean; LLC vs Limited Liability Company; registration certificate as clean PNG scan", c,
                    scans={"registration_certificate": ("clean", "png")}, expected=E(
                        notes="Form says 'Limited Liability Company'; certificate and bank say 'LLC'.")))

    c = us_company(g, "Granite Peak Tools LLC", ADDR_US[2], "Texas")
    apps.append(App("US-003", "US", "Edge 1: wrong document attached in the W-9 slot (it is a bank letter)", c,
                    wrong_slot={"tax_certificate": "bank_proof"}, expected=E(
                        edge_case="1 Missing information (wrong document)", demo_priority="1", wrong_type_documents="tax_certificate",
                        missing_documents="tax_certificate", auto_decision_demo="Pending",
                        reason_codes="WRONG_DOCUMENT_TYPE;MISSING_DOCUMENT", required_actions="Upload a completed Form W-9",
                        notes="Declared-type completeness passes; detected-type completeness fails after extraction.")))

    c = us_company(g, "Cobalt Ridge Systems Inc.", ADDR_US[3], "Ohio", "C Corporation")
    c["ein"] = c["ein"][:-1]                                   # 8 digits instead of 9
    apps.append(App("US-004", "US", "Edge 3: EIN has 8 digits instead of 9", c, expected=E(
        edge_case="3 Invalid tax ID", demo_priority="2", auto_decision_demo="Pending", reason_codes="INVALID_FORMAT",
        required_actions="Correct the EIN (format NN-NNNNNNN, 9 digits)")))

    c = us_company(g, "Maple Street Bakery Supply LLC", ADDR_US[4], "Illinois")
    r = c["routing_number"]
    c["routing_number"] = r[:-1] + str((int(r[-1]) + 3) % 10)    # breaks the 3-7-1 checksum
    apps.append(App("US-005", "US", "Edge 3: ABA routing number fails its checksum", c, expected=E(
        edge_case="3 Invalid bank identifier", auto_decision_demo="Pending", reason_codes="INVALID_FORMAT",
        required_actions="Correct the 9-digit ABA routing number")))

    c = us_company(g, "Summit Crest Consulting LLC", ADDR_US[5], "Idaho")
    c["bank_holder_name"] = c["bank_proof_holder"] = "Blue Heron Holdings LLC"
    apps.append(App("US-006", "US", "Edge 2: bank account holder is a different company", c, expected=E(
        edge_case="2 Bank/company name mismatch", demo_priority="3", auto_decision_demo="Pending", reason_codes="NAME_MISMATCH_BANK",
        required_actions="Clarify the relationship to the account holder; routed for manual review")))

    c = us_company(g, "Prairie Wind Data Services LLC", ADDR_US[6], "Colorado")
    c["ein"] = us001["ein"]
    apps.append(App("US-007", "US", "Edge 4: same EIN as US-001", c, expected=E(
        edge_case="4 Duplicate vendor", demo_priority="4", auto_decision_demo="Pending", reason_codes="DUPLICATE_SUSPECTED",
        required_actions="Reviewer confirms whether this is the same vendor; US-001 is referenced", notes="Process US-001 first.")))

    c = us_company(g, "Harbor Light Marine LLC", ADDR_US[7], "Florida")
    apps.append(App("US-008", "US", "Edge 5: simulated business-registry lookup times out", c, expected=E(
        edge_case="5 API failure", auto_decision_demo="Pending", reason_codes="VERIFICATION_UNAVAILABLE",
        required_actions="Retry verification later or complete manual review; verification is NOT treated as failed")))

    c = us_company(g, "Ironwood Fabrication Inc.", ADDR_US[8], "Pennsylvania", "C Corporation")
    apps.append(App("US-009", "US", "Edge 6: poor-quality scan of the W-9", c, scans={"tax_certificate": ("poor", "pdf")}, expected=E(
        edge_case="6 Low-confidence extraction", auto_decision_demo="Pending", reason_codes="LOW_CONFIDENCE_EXTRACTION",
        required_actions="Reviewer verifies the EIN against the document image and reprocesses",
        notes="Acceptable outcomes: correct value OR flagged LOW.")))

    c = us_company(g, "Falcon Crest Trading LLC", ADDR_US[9], "Arizona")
    apps.append(App("US-010", "US", "Documented rejection scenario: EIN is on a synthetic internal blocklist", c, expected=E(
        edge_case="Rejection (R1)", demo_priority="", auto_decision_demo="Pending", reason_codes="BLOCKLIST_MATCH",
        final_after_review="Rejected (R1: reviewer confirms condition DQ-01 with evidence)",
        required_actions="Reviewer confirms or clears the blocklist match",
        notes="Rejected is reached ONLY via a human reviewer confirming a documented synthetic condition (see internal_blocklist.json).")))

    # --------------------------------------------------------------------- UK
    c = gb_company(g, "Thistle & Oak Design Limited", ADDR_GB[0])
    gb001 = c
    apps.append(App("UK-001", "GB", "Clean, complete application", c, expected=E(production_decision="Pending (VERIFICATION_NOT_LIVE)")))

    c = gb_company(g, "Marsh and Fenn Limited", ADDR_GB[8], True)
    c["legal_name"] = c["trade_name"] = "Marsh & Fenn Ltd"
    c["bank_holder_name"] = c["bank_proof_holder"] = "MARSH & FENN LTD"
    apps.append(App("UK-002", "GB", "Clean; '&' vs 'and', Ltd vs Limited; scanned bank proof (clean scan)", c,
                    scans={"bank_proof": ("clean", "pdf")}, expected=E(notes="Certificate says 'Marsh and Fenn Limited'.")))

    c = gb_company(g, "Ashdown Packaging Limited", ADDR_GB[4])
    apps.append(App("UK-003", "GB", "Edge 1: certificate of incorporation missing", c, drop_docs=["registration_certificate"], expected=E(
        edge_case="1 Missing information", demo_priority="1", missing_documents="registration_certificate", auto_decision_demo="Pending",
        reason_codes="MISSING_DOCUMENT", required_actions="Upload the certificate of incorporation")))

    c = gb_company(g, "Pennine Metals Limited", ADDR_GB[5])
    v = c["vat_number"][2:]
    bad = v[:8] + str((int(v[8]) + 1) % 10)
    c["vat_number"] = "GB" + bad
    c["vat_display"] = f"GB {bad[:3]} {bad[3:7]} {bad[7:]}"
    apps.append(App("UK-004", "GB", "Edge 3: VAT number fails the mod-97 check", c, expected=E(
        edge_case="3 Invalid tax ID", demo_priority="2", auto_decision_demo="Pending", reason_codes="INVALID_FORMAT",
        required_actions="Correct the VAT number (GB + 9 digits, valid check digits)")))

    c = gb_company(g, "Harrow Bridge Services Limited", ADDR_GB[3])
    c["bank_holder_name"] = c["bank_proof_holder"] = "Westgate Ventures Limited"
    apps.append(App("UK-005", "GB", "Edge 2: bank account holder is a different company", c, expected=E(
        edge_case="2 Bank/company name mismatch", demo_priority="3", auto_decision_demo="Pending", reason_codes="NAME_MISMATCH_BANK",
        required_actions="Clarify the relationship to the account holder; routed for manual review")))

    c = gb_company(g, "Thistle and Oak Studio Limited", ADDR_GB[2])
    c["company_number"] = gb001["company_number"]
    apps.append(App("UK-006", "GB", "Edge 4: same Companies House number as UK-001", c, expected=E(
        edge_case="4 Duplicate vendor", demo_priority="4", auto_decision_demo="Pending", reason_codes="DUPLICATE_SUSPECTED",
        required_actions="Reviewer confirms whether this is the same vendor; UK-001 is referenced", notes="Process UK-001 first.")))

    c = gb_company(g, "Calder Valley Energy Limited", ADDR_GB[6])
    apps.append(App("UK-007", "GB", "Edge 5: simulated HMRC VAT lookup returns HTTP 503 persistently", c, expected=E(
        edge_case="5 API failure", auto_decision_demo="Pending", reason_codes="VERIFICATION_UNAVAILABLE",
        required_actions="Retry verification later or complete manual review; verification is NOT treated as failed")))

    c = gb_company(g, "Quillon Instruments Limited", ADDR_GB[9])
    apps.append(App("UK-008", "GB", "Edge 6: poor-quality scan of the VAT certificate", c, scans={"tax_certificate": ("poor", "pdf")},
                    expected=E(edge_case="6 Low-confidence extraction", auto_decision_demo="Pending", reason_codes="LOW_CONFIDENCE_EXTRACTION",
                               required_actions="Reviewer verifies the VAT number against the document image and reprocesses",
                               notes="Acceptable outcomes: correct value OR flagged LOW.")))

    c = gb_company(g, "Fernside Crafts Limited", ADDR_GB[7])
    c["vat_number"] = c["vat_display"] = ""
    c["declarations"] = {"vat_not_registered": True}
    apps.append(App("UK-009", "GB", "Vendor declares it is not VAT-registered (no VAT number or certificate)", c, drop_docs=["tax_certificate"],
                    expected=E(auto_decision_demo="Pending", reason_codes="DECLARATION_NEEDS_REVIEW",
                               final_after_review="Approved (DEMO) after reviewer accepts the non-registration declaration",
                               required_actions="Reviewer accepts or rejects the declaration of non-registration",
                               notes="Completeness passes (declaration present); the declaration itself needs human acceptance in Phase 4.")))
    return apps


# ============================================================================= assembly
def required_doc_types(country: str) -> List[str]:
    return {"IN": ["registration_form", "registration_certificate", "tax_certificate", "tax_id_proof", "bank_proof"],
            "US": ["registration_form", "registration_certificate", "tax_certificate", "bank_proof"],
            "GB": ["registration_form", "registration_certificate", "tax_certificate", "bank_proof"]}[country]


def build_doc(country: str, c: Dict[str, Any], dtype: str) -> DocSpec:
    return {"registration_form": lambda: doc_form(country, c), "registration_certificate": lambda: doc_reg_cert(country, c),
            "tax_certificate": lambda: doc_tax_cert(country, c), "tax_id_proof": lambda: doc_pan_proof(c),
            "bank_proof": lambda: doc_bank_proof(country, c)}[dtype]()


def main() -> None:
    g = Gen(SEED)
    apps = build_apps(g)
    docs_root = OUT / "documents"
    if docs_root.exists():
        shutil.rmtree(docs_root)
    docs_root.mkdir(parents=True)

    applications, truth, manifest_rows = [], {}, []
    fixtures: Dict[str, Any] = {"_comment": "SIMULATED provider data. Used only by simulated adapters; never real verification.",
                                "IN": {"gstin": {}, "pan": {}, "bank": {}, "fault": {}},
                                "US": {"registry": {}, "bank": {}, "tin_match": {}, "fault": {}},
                                "GB": {"companies_house": {}, "hmrc_vat": {}, "bank": {}, "fault": {}}}
    blocklist = {"_comment": "FICTIONAL internal blocklist for the documented rejection scenario. Not real data.",
                 "conditions": {"DQ-01": "Previously terminated for submitting falsified bank letters (fictional internal record)"},
                 "entries": []}

    for i, app in enumerate(apps):
        c = finalize(app.company)
        country = app.country
        folder = docs_root / app.ref
        folder.mkdir()
        doc_entries, doc_truth = [], {}
        for dtype in required_doc_types(country):
            if dtype in app.drop_docs:
                continue
            content_type = app.wrong_slot.get(dtype, dtype)
            spec = build_doc(country, c, content_type)
            scan, fmt = app.scans.get(content_type, ("none", "pdf"))
            spec.scan, spec.fmt = scan, fmt
            spec.filename = f"{dtype}.{fmt}" if not app.wrong_slot.get(dtype) else f"{dtype}.pdf"
            path = write_document(spec, folder, app.ref, SEED + i * 17 + len(spec.filename))
            doc_entries.append({"type": dtype, "path": f"documents/{app.ref}/{path.name}"})
            doc_truth[path.name] = {"declared_type": dtype, "actual_type": content_type, "scan": scan,
                                    "fields": {f: v for _, v, f in spec.lines if f}}
        applications.append({"ref": app.ref, "country": country, "scenario": app.scenario, "fields": c["form"],
                             "declarations": c["declarations"], "documents": doc_entries})
        truth[app.ref] = doc_truth

        exp = app.expected
        manifest_rows.append({"ref": app.ref, "country": country, "scenario": app.scenario, "edge_case": exp["edge_case"],
                              "demo_priority": exp["demo_priority"], "documents": ";".join(d["type"] for d in doc_entries),
                              "scanned_documents": ";".join(f"{k}:{v[0]}" for k, v in app.scans.items()),
                              "expected_missing_fields": exp["missing_fields"], "expected_missing_documents": exp["missing_documents"],
                              "expected_wrong_type_documents": exp["wrong_type_documents"],
                              "expected_auto_decision_demo": exp["auto_decision_demo"], "expected_reason_codes": exp["reason_codes"],
                              "expected_final_after_review": exp["final_after_review"],
                              "expected_production_decision": exp["production_decision"],
                              "required_actions": exp["required_actions"], "notes": exp["notes"]})

        # --- simulated-provider fixtures (first application owning an identifier wins)
        if country == "IN":
            if c["gstin"] and c["gstin"] not in fixtures["IN"]["gstin"]:
                fixtures["IN"]["gstin"][c["gstin"]] = {"legal_name": c["cert_name"], "trade_name": c["cert_trade_name"],
                                                       "status": "CANCELLED" if app.ref == "IN-010" else "ACTIVE"}
            fixtures["IN"]["pan"].setdefault(c["pan"], {"name": c["cert_name"]})
            fixtures["IN"]["bank"].setdefault(c["bank_account_number"], {"holder_name": c["bank_proof_holder"]})
        elif country == "US":
            key = f"{c['state']}:{c['state_file_number']}"
            fixtures["US"]["registry"].setdefault(key, {"name": c["cert_name"], "status": "ACTIVE"})
            fixtures["US"]["tin_match"].setdefault(c["ein"].replace("-", ""), {"name": c["cert_name"], "match": True})
            fixtures["US"]["bank"].setdefault(c["bank_account_number"], {"holder_name": c["bank_proof_holder"]})
            if app.ref == "US-008":
                fixtures["US"]["fault"][key] = "TIMEOUT"
            if app.ref == "US-010":
                blocklist["entries"].append({"id_type": "EIN", "value_norm": c["ein"].replace("-", ""), "condition_code": "DQ-01",
                                             "record_ref": "SYN-BL-0001", "added": "2026-02-14"})
        else:
            fixtures["GB"]["companies_house"].setdefault(c["company_number"], {"name": c["cert_name"], "status": "active"})
            if c["vat_number"]:
                fixtures["GB"]["hmrc_vat"].setdefault(c["vat_number"][2:], {"name": c["cert_name"]})
            fixtures["GB"]["bank"].setdefault(c["bank_account_number"], {"holder_name": c["bank_proof_holder"]})
            if app.ref == "UK-007":
                fixtures["GB"]["fault"][c["vat_number"][2:]] = "HTTP_503"

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "applications.json").write_text(json.dumps(applications, indent=2), encoding="utf-8")
    (OUT / "ground_truth.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")
    (OUT / "simulation_fixtures.json").write_text(json.dumps(fixtures, indent=2), encoding="utf-8")
    (OUT / "internal_blocklist.json").write_text(json.dumps(blocklist, indent=2), encoding="utf-8")
    with open(OUT / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(manifest_rows[0].keys()))
        w.writeheader()
        w.writerows(manifest_rows)
    n_docs = sum(len(t) for t in truth.values())
    print(f"Generated {len(applications)} applications, {n_docs} documents -> {OUT}")


if __name__ == "__main__":
    main()
