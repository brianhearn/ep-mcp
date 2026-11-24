"""Pack-level integrity verification (W-INT-01..02), security lint (W-SEC-01..05),
and shipped-vector detection for ep-mcp.

Canonical pack_digest and security patterns mirror
``expert-pack/tools/validator/ep-validate.py`` so spec validation and runtime
indexing cannot silently diverge.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Must stay byte-for-byte compatible with expert-pack/tools/validator/ep-validate.py
INTEGRITY_SKIP_DIRS = frozenset({
    ".obsidian", ".git", ".ep-mcp", "node_modules", "__pycache__", ".venv",
})
INTEGRITY_SKIP_FILES = frozenset({"manifest.yaml"})
TEXT_EXTS = frozenset({".md", ".yaml", ".yml", ".json", ".txt"})

# Shipped precomputed vector/index artifacts that MUST be ignored (always re-embed from Markdown)
SHIPPED_VECTOR_EXTS = frozenset({
    ".vec", ".faiss", ".npy", ".npz", ".lance", ".hnsw", ".ann", ".db", ".sqlite", ".sqlite3",
})

# ── Security lint patterns (W-SEC-01, W-SEC-02, W-SEC-03, W-SEC-05) ──────────

RE_FENCED_BLOCK = re.compile(r"^(?:```|~~~).*?^(?:```|~~~)\s*$", re.MULTILINE | re.DOTALL)
RE_INLINE_CODE = re.compile(r"`[^`\n]+`")

RE_HIDDEN_UNICODE = re.compile(
    r"[\u200b-\u200f\u202a-\u202e\u2066-\u2069\ufeff\U000e0000-\U000e007f]"
)

SEC_OVERRIDE_PATTERNS = [
    (
        "prompt-override",
        re.compile(
            r"\b(?:ignore|disregard|forget)\s+(?:all\s+|any\s+)?"
            r"(?:previous|prior|above|earlier|system|developer|safety)\s+"
            r"(?:instructions|prompts|rules|directives|constraints|guardrails)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "role-hijack",
        re.compile(
            r"\b(?:you\s+must\s+now\s+act\s+as|override\s+system\s+prompt|"
            r"new\s+system\s+instructions\s*:|enter\s+developer\s+mode)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "outbound-exfil",
        re.compile(
            r"\b(?:exfiltrate|send|post|upload|transmit|forward|curl|wget|fetch)\s+"
            r"(?:the\s+|all\s+|any\s+|user\s+|your\s+|session\s+)?"
            r"(?:api[_ -]?keys?|secrets?|tokens?|passwords?|credentials?|"
            r"ssh[_ -]?keys?|env(?:ironment)?\s+variables?|system\s+prompt|"
            r"conversation\s+history|session\s+state)\s+"
            r"(?:to|via)\s+(?:https?://|\S*webhook|\S*endpoint)",
            re.IGNORECASE,
        ),
    ),
    (
        "image-beacon",
        re.compile(
            r"!\[[^\]]*\]\(https?://[^)\s]+\?[^)\s]*"
            r"(?:\$\{[^}]+\}|\{\{[^}]+\}\}|\$[A-Z_][A-Z0-9_]{2,}|"
            r"(?:api[_-]?key|secret|token|password|env)=)",
            re.IGNORECASE,
        ),
    ),
]

SECRET_PATTERNS = [
    ("openai-or-anthropic-key", re.compile(r"\bsk-(?:proj-|ant-)?[A-Za-z0-9_\-]{20,}\b")),
    ("github-pat", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b")),
    ("slack-token", re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}\b")),
    ("aws-access-key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("private-key-block", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |PRIVATE )?PRIVATE KEY-----")),
    (
        "credential-assignment",
        re.compile(
            r"(?:api[_-]?key|secret[_-]?key|access[_-]?token|auth[_-]?token|"
            r"client[_-]?secret|private[_-]?key)\s*[:=]\s*"
            r"[\"']?(?!(?:<REDACTED>|REDACTED|YOUR_|your_|example|placeholder|"
            r"xxxx|XXXX|\*\*\*\*|\.\.\.|null|none|true|false|\$\{?))[A-Za-z0-9_\-\.~+/]{20,}[\"']?",
            re.IGNORECASE,
        ),
    ),
]

PLACEHOLDER_SECRET_RE = re.compile(
    r"^(?:sk-[xX0-9]+|sk-(?:proj-|ant-)?(?:your|example|test|dummy|placeholder|xxxx)[A-Za-z0-9_\-]*|"
    r"gh[pousr]_[xX0]+|AKIA[0X]+|.*(?:example|placeholder|redacted|dummy|sample|test_key|your_key|"
    r"X{4,}|x{4,}|0{6,}|\*{4,}).*)$",
    re.IGNORECASE,
)

OVERSIGHT_LOOSENING_PATTERNS = [
    (
        "skip-confirmation-flag",
        re.compile(
            r"(?:--no-verify|--skip-verify|--force-yes|--assume-yes|"
            r"--dangerously-skip-permissions|\b--yes\b)",
            re.IGNORECASE,
        ),
    ),
    (
        "skip-oversight-phrase",
        re.compile(
            r"\b(?:without\s+(?:asking(?:\s+(?:the\s+)?user)?|confirmation|user\s+confirmation|"
            r"confirming\s+with\s+(?:the\s+)?(?:user|human|owner|operator)|"
            r"approval|prompting\s+(?:the\s+)?(?:user|for\s+confirmation)|human\s+review)|"
            r"skip\s+(?:confirmation|approval|human\s+review|verification|safety\s+checks?)|"
            r"bypass\s+(?:confirmation|approval|review|safety\s+checks?|hooks?)|"
            r"never\s+ask\s+(?:for\s+)?(?:confirmation|permission|approval)|"
            r"auto[- ]?approve\s+all|"
            r"proceed\s+without\s+(?:asking|confirmation))\b",
            re.IGNORECASE,
        ),
    ),
]

OVERSIGHT_GUARD_RE = re.compile(
    r"\b(?:never\s+(?:use|pass|run|add|skip|bypass)|do\s+not|don't|avoid|forbid|prohibit|refuse|must\s+not|"
    r"anti[- ]?pattern|gotcha|mistake|dangerous|unsafe|"
    r"only\s+(?:in|when|if|after|for)|"
    r"in\s+(?:ci|test|testing|sandbox|staging|non[- ]?prod|local\s+dev)\s+only|"
    r"after\s+(?:tests?\s+pass|approval|confirmation|explicit\s+consent)|"
    r"when\s+(?:explicitly\s+)?(?:approved|requested|authorized|confirmed)|"
    r"if\s+(?:explicitly\s+)?(?:approved|requested|authorized|confirmed)|"
    r"with\s+explicit\s+(?:user\s+)?(?:approval|permission|consent)|"
    r"except\s+when)\b",
    re.IGNORECASE,
)


class PackSecurityError(Exception):
    """Raised when a pack fails integrity or security policy at index/load time."""


@dataclass
class IntegrityReport:
    """Result of verifying a pack's manifest.yaml integrity block."""

    status: str = "unsigned"  # unsigned | verified | unverifiable
    digest_verified: bool = False
    declared_digest: str | None = None
    actual_digest: str = ""
    issues: list[str] = field(default_factory=list)


@dataclass
class SecurityFinding:
    """A single W-SEC-* lint finding in a pack file."""

    code: str       # W-SEC-01 | W-SEC-02 | W-SEC-03 | W-SEC-05
    file_path: str  # pack-relative POSIX path
    message: str


def compute_pack_digest(pack_dir: str | Path) -> str:
    """Compute canonical SHA-256 digest over all distributed pack files.

    Matches ``expert-pack/tools/validator/ep-validate.py::compute_pack_digest``.
    """
    pack_path = os.path.abspath(str(pack_dir))
    records: list[tuple[str, str]] = []
    for root, dirs, files in os.walk(pack_path):
        rel_root = os.path.relpath(root, pack_path).replace(os.sep, "/")
        if rel_root == ".":
            rel_root = ""
        dirs[:] = sorted(
            d for d in dirs
            if d not in INTEGRITY_SKIP_DIRS
            and not (rel_root == "eval" and d == "results")
        )
        for fname in sorted(files):
            if fname in INTEGRITY_SKIP_FILES and not rel_root:
                continue
            if fname in {".DS_Store", "Thumbs.db"}:
                continue
            full = os.path.join(root, fname)
            rel = os.path.relpath(full, pack_path).replace(os.sep, "/")
            if rel.startswith("eval/results/"):
                continue
            ext = os.path.splitext(fname)[1].lower()
            try:
                with open(full, "rb") as fh:
                    raw = fh.read()
            except OSError:
                continue
            if ext in TEXT_EXTS:
                raw = raw.replace(b"\r\n", b"\n")
            file_hash = hashlib.sha256(raw).hexdigest()
            records.append((rel, file_hash))
    records.sort(key=lambda item: item[0])
    manifest_lines = "".join(f"{rel}\tsha256:{h}\n" for rel, h in records)
    return "sha256:" + hashlib.sha256(manifest_lines.encode("utf-8")).hexdigest()


def verify_pack_signature(
    payload_bytes: bytes,
    signature_str: str | None,
    public_key_str: str | None = None,
    signed_by_str: str | None = None,
) -> tuple[bool, str]:
    """Verify an optional signature over '<slug>@<pack_version>:<pack_digest>'."""
    sig_s = str(signature_str or "").strip()
    pub_s = str(public_key_str or "").strip()
    if not sig_s:
        return False, "empty signature"
    if not pub_s:
        return False, "missing public_key (cannot verify signature without public key)"

    if "-----BEGIN SSH SIGNATURE-----" in sig_s and pub_s.startswith("ssh-"):
        ssh_keygen = shutil.which("ssh-keygen")
        if not ssh_keygen:
            return False, "ssh-keygen not found on PATH to verify SSH signature"
        signer_id = str(signed_by_str or "expertpack").strip() or "expertpack"
        try:
            with tempfile.TemporaryDirectory(prefix="ep-sig-") as tmp:
                allowed = os.path.join(tmp, "allowed_signers")
                sig_file = os.path.join(tmp, "pack.sig")
                with open(allowed, "w", encoding="utf-8") as f:
                    f.write(f"{signer_id} {pub_s}\n")
                with open(sig_file, "w", encoding="utf-8") as f:
                    f.write(sig_s + "\n")
                proc = subprocess.run(
                    [
                        ssh_keygen, "-Y", "verify", "-f", allowed, "-I", signer_id,
                        "-n", "expertpack", "-s", sig_file,
                    ],
                    input=payload_bytes,
                    capture_output=True,
                    timeout=10,
                )
                if proc.returncode == 0:
                    return True, "verified via ssh-keygen"
                err = (proc.stderr or proc.stdout or b"").decode("utf-8", errors="replace").strip()
                return False, f"ssh-keygen verification failed: {err or 'invalid signature'}"
        except Exception as exc:
            return False, f"ssh-keygen verification error: {exc}"

    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except ImportError:
        return False, "cryptography package not installed to verify Ed25519 signature"

    try:
        raw_sig = sig_s
        if raw_sig.startswith("ed25519:"):
            raw_sig = raw_sig[len("ed25519:"):].strip()
        try:
            sig_bytes = (
                bytes.fromhex(raw_sig)
                if re.fullmatch(r"[0-9a-fA-F]{128}", raw_sig)
                else base64.b64decode(raw_sig, validate=True)
            )
        except Exception:
            return False, "malformed signature encoding (expected hex or base64)"

        if pub_s.startswith("ed25519:"):
            key_body = pub_s[len("ed25519:"):].strip()
            pub_bytes = (
                bytes.fromhex(key_body)
                if re.fullmatch(r"[0-9a-fA-F]{64}", key_body)
                else base64.b64decode(key_body, validate=True)
            )
            pub_key = Ed25519PublicKey.from_public_bytes(pub_bytes)
        elif pub_s.startswith("ssh-ed25519 "):
            pub_key = serialization.load_ssh_public_key(pub_s.encode("utf-8"))
        elif "-----BEGIN PUBLIC KEY-----" in pub_s:
            pub_key = serialization.load_pem_public_key(pub_s.encode("utf-8"))
        else:
            pub_bytes = (
                bytes.fromhex(pub_s)
                if re.fullmatch(r"[0-9a-fA-F]{64}", pub_s)
                else base64.b64decode(pub_s, validate=True)
            )
            pub_key = Ed25519PublicKey.from_public_bytes(pub_bytes)

        pub_key.verify(sig_bytes, payload_bytes)
        return True, "verified via Ed25519"
    except Exception as exc:
        return False, f"signature verification failed: {exc}"


def verify_pack_integrity(
    pack_dir: str | Path,
    manifest_raw: dict[str, Any],
    slug: str | None = None,
) -> IntegrityReport:
    """Verify optional manifest.yaml integrity block for a pack."""
    actual_digest = compute_pack_digest(pack_dir)
    block = manifest_raw.get("integrity")
    if block is None:
        return IntegrityReport(
            status="unsigned",
            digest_verified=False,
            declared_digest=None,
            actual_digest=actual_digest,
        )

    issues: list[str] = []
    if not isinstance(block, dict):
        return IntegrityReport(
            status="unverifiable",
            digest_verified=False,
            actual_digest=actual_digest,
            issues=["W-INT-01: integrity block must be a mapping"],
        )

    algo = str(block.get("algorithm") or "sha256").strip().lower()
    if algo != "sha256":
        return IntegrityReport(
            status="unverifiable",
            digest_verified=False,
            actual_digest=actual_digest,
            issues=[f"W-INT-01: unsupported integrity.algorithm '{algo}'"],
        )

    manifest_ver = str(manifest_raw.get("version") or "").strip()
    sealed_ver = str(block.get("pack_version") or "").strip()
    declared_digest = str(block.get("pack_digest") or "").strip()

    digest_ok = True
    if not sealed_ver:
        digest_ok = False
        issues.append("W-INT-01: integrity.pack_version is missing")
    elif manifest_ver and sealed_ver != manifest_ver:
        digest_ok = False
        issues.append(
            f"W-INT-01: integrity.pack_version '{sealed_ver}' != manifest.version '{manifest_ver}'"
        )

    if not declared_digest:
        digest_ok = False
        issues.append("W-INT-01: integrity.pack_digest is missing")
    elif declared_digest != actual_digest:
        digest_ok = False
        issues.append(
            f"W-INT-01: integrity.pack_digest mismatch "
            f"(declared {declared_digest[:23]}..., actual {actual_digest[:23]}...)"
        )

    signature = str(block.get("signature") or "").strip()
    if not signature:
        return IntegrityReport(
            status="unsigned" if digest_ok else "unverifiable",
            digest_verified=digest_ok,
            declared_digest=declared_digest or None,
            actual_digest=actual_digest,
            issues=issues,
        )

    if not digest_ok:
        return IntegrityReport(
            status="unverifiable",
            digest_verified=False,
            declared_digest=declared_digest or None,
            actual_digest=actual_digest,
            issues=issues,
        )

    pack_slug = slug or str(manifest_raw.get("slug") or "").strip()
    payload = f"{pack_slug}@{sealed_ver}:{declared_digest}".encode("utf-8")
    ok, reason = verify_pack_signature(
        payload,
        signature_str=signature,
        public_key_str=block.get("public_key"),
        signed_by_str=block.get("signed_by"),
    )
    if ok:
        return IntegrityReport(
            status="verified",
            digest_verified=True,
            declared_digest=declared_digest,
            actual_digest=actual_digest,
            issues=[],
        )

    issues.append(f"W-INT-02: integrity.signature is unverifiable ({reason})")
    return IntegrityReport(
        status="unverifiable",
        digest_verified=True,
        declared_digest=declared_digest,
        actual_digest=actual_digest,
        issues=issues,
    )


def _strip_code_for_prose_scan(text: str) -> str:
    cleaned = RE_FENCED_BLOCK.sub("\n", text)
    cleaned = RE_INLINE_CODE.sub(" ", cleaned)
    return cleaned


def _find_unscoped_oversight_loosening(text: str) -> tuple[str, str] | None:
    stripped = RE_FENCED_BLOCK.sub("\n", text)
    if stripped.startswith("---"):
        end = stripped.find("\n---", 3)
        if end != -1:
            stripped = stripped[end + 4:]
    for line in stripped.splitlines():
        s = line.strip()
        if not s:
            continue
        for kind, pat in OVERSIGHT_LOOSENING_PATTERNS:
            m = pat.search(s)
            if m and not OVERSIGHT_GUARD_RE.search(s):
                return kind, " ".join(s.split())[:90]
    return None


def scan_text_security(rel_path: str, text: str, ext: str = ".md") -> list[SecurityFinding]:
    """Run W-SEC-01, W-SEC-02, W-SEC-03, W-SEC-05 over one text file."""
    findings: list[SecurityFinding] = []

    non_fenced = RE_FENCED_BLOCK.sub("\n", text)
    check_uni = non_fenced[1:] if non_fenced.startswith("\ufeff") else non_fenced
    m_uni = RE_HIDDEN_UNICODE.search(check_uni)
    if m_uni:
        ch = m_uni.group(0)
        findings.append(SecurityFinding(
            code="W-SEC-01",
            file_path=rel_path,
            message=f"Hidden/bidi/tag Unicode character (U+{ord(ch):04X}) outside code fences",
        ))

    if ext == ".md":
        prose = _strip_code_for_prose_scan(text)
        for kind, pat in SEC_OVERRIDE_PATTERNS:
            m_sec = pat.search(prose)
            if m_sec:
                snippet = " ".join(m_sec.group(0).split())[:80]
                findings.append(SecurityFinding(
                    code="W-SEC-02",
                    file_path=rel_path,
                    message=f"Suspicious {kind} directive in prose: '{snippet}'",
                ))
                break

    for kind, pat in SECRET_PATTERNS:
        m_secret = pat.search(text)
        if m_secret:
            matched_val = m_secret.group(0).strip()
            if PLACEHOLDER_SECRET_RE.match(matched_val):
                continue
            preview = matched_val[:24] + ("..." if len(matched_val) > 24 else "")
            findings.append(SecurityFinding(
                code="W-SEC-03",
                file_path=rel_path,
                message=f"Potential secret ({kind}) detected: '{preview}'",
            ))
            break

    if ext == ".md":
        hit = _find_unscoped_oversight_loosening(text)
        if hit:
            kind, snippet = hit
            findings.append(SecurityFinding(
                code="W-SEC-05",
                file_path=rel_path,
                message=f"Unscoped oversight-loosening directive ({kind}): '{snippet}'",
            ))

    return findings


def scan_pack_security(
    pack_dir: str | Path,
    manifest_raw: dict[str, Any] | None = None,
) -> list[SecurityFinding]:
    """Scan all text files and manifest mcp.instructions in a pack for W-SEC-01/02/03/05."""
    findings: list[SecurityFinding] = []
    pack_path = os.path.abspath(str(pack_dir))

    for root, dirs, files in os.walk(pack_path):
        dirs[:] = sorted(d for d in dirs if d not in INTEGRITY_SKIP_DIRS)
        for fname in sorted(files):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in TEXT_EXTS:
                continue
            full = os.path.join(root, fname)
            rel = os.path.relpath(full, pack_path).replace(os.sep, "/")
            try:
                with open(full, encoding="utf-8") as fh:
                    text = fh.read()
            except (OSError, UnicodeDecodeError):
                continue
            findings.extend(scan_text_security(rel, text, ext=ext))

    if isinstance(manifest_raw, dict):
        mcp_block = manifest_raw.get("mcp")
        if isinstance(mcp_block, dict):
            instr = str(mcp_block.get("instructions") or "")
            if instr:
                for kind, pat in SEC_OVERRIDE_PATTERNS:
                    m_sec = pat.search(instr)
                    if m_sec:
                        snippet = " ".join(m_sec.group(0).split())[:80]
                        findings.append(SecurityFinding(
                            code="W-SEC-02",
                            file_path="manifest.yaml",
                            message=f"Suspicious {kind} directive in mcp.instructions: '{snippet}'",
                        ))
                        break
                hit = _find_unscoped_oversight_loosening(instr)
                if hit:
                    kind, snippet = hit
                    findings.append(SecurityFinding(
                        code="W-SEC-05",
                        file_path="manifest.yaml",
                        message=f"Unscoped oversight-loosening directive ({kind}) in mcp.instructions: '{snippet}'",
                    ))

    return findings


def detect_shipped_vector_files(pack_dir: str | Path) -> list[str]:
    """Find precomputed vector/embedding files shipped inside a pack directory.

    Excludes ``.ep-mcp/`` (the server's own local index) and infrastructure dirs.
    Any returned files MUST be ignored — ep-mcp always re-embeds from Markdown.
    """
    pack_path = os.path.abspath(str(pack_dir))
    found: list[str] = []
    for root, dirs, files in os.walk(pack_path):
        dirs[:] = sorted(d for d in dirs if d not in INTEGRITY_SKIP_DIRS)
        for fname in sorted(files):
            ext = os.path.splitext(fname)[1].lower()
            if ext in SHIPPED_VECTOR_EXTS:
                full = os.path.join(root, fname)
                rel = os.path.relpath(full, pack_path).replace(os.sep, "/")
                found.append(rel)
    return found
