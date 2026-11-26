"""MegaRL brand kit integration.

Source of truth is the vendored kit in `brand/`:
  - `brand/design-tokens.css` — palette tokens (parsed at runtime)
  - `brand/assets/*.svg` — editable vector marks (inlined, not re-drawn)
  - `brand/BRAND-GUIDELINES.md` — usage rules (dark lockup, endorsement, tagline)

Fonts are not bundled by the kit; the specified Inter Display / Inter stack is
used with system fallbacks.
"""

from __future__ import annotations

import base64
import re
from functools import lru_cache
from pathlib import Path

BRAND_DIR = Path(__file__).resolve().parents[2] / "brand"
ASSETS = BRAND_DIR / "assets"
TOKENS_CSS = (BRAND_DIR / "design-tokens.css").read_text(encoding="utf-8")


def _token(name: str) -> str:
    match = re.search(rf"--megarl-{name}:\s*(#[0-9A-Fa-f]{{6}})", TOKENS_CSS)
    if not match:
        raise RuntimeError(f"missing brand token --megarl-{name}")
    return match.group(1)


BRAND = {
    "navy": _token("navy"),
    "citron": _token("citron"),
    "cyan": _token("cyan"),
    "orange": _token("orange"),
    "white": _token("white"),
    "muted": _token("muted"),
}

TAGLINE = "Learning in motion."
ENDORSEMENT = "by TensorPlay"


@lru_cache(maxsize=None)
def _asset(name: str) -> str:
    return (ASSETS / name).read_text(encoding="utf-8")


def lockup_svg(*, dark: bool = True) -> str:
    name = "logo-horizontal-on-dark.svg" if dark else "logo-horizontal-on-light.svg"
    return _asset(name)


def symbol_svg() -> str:
    return _asset("logo-symbol-color.svg")


def favicon_uri() -> str:
    payload = (ASSETS / "favicon.svg").read_bytes()
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def brand_css() -> str:
    return (
        TOKENS_CSS
        + """
        :root{color-scheme:dark;}
        body{margin:0;background:var(--megarl-navy);color:var(--megarl-white);
             font-family:'Inter Display','Inter',system-ui,-apple-system,sans-serif;}
        h1,h2{font-weight:650;letter-spacing:-0.03em;}
        h1 span{color:var(--megarl-citron);}
        .tp-nav{height:68px;display:flex;align-items:center;gap:1rem;
                padding:0 1.5rem;border-bottom:1px solid rgba(170,191,199,.25);}
        .tp-brand svg{height:36px;width:auto;display:block;}
        .tp-tag{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;
                color:var(--megarl-muted);letter-spacing:.02em;}
        .tp-tag b{color:var(--megarl-citron);font-weight:500;}
        .card{background:rgba(245,248,250,.04);border:1px solid rgba(170,191,199,.22);
              border-radius:12px;padding:1.1rem 1.4rem;margin:1rem 0;}
        .card b{color:var(--megarl-white);}
        table{border-collapse:collapse;width:100%;font-size:14px;}
        td{padding:.35rem 1rem;border-bottom:1px solid rgba(170,191,199,.15);}
        td:last-child{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
                      color:var(--megarl-citron);}
        footer{padding:2rem 1.5rem;color:var(--megarl-muted);font-size:12px;
               border-top:1px solid rgba(170,191,199,.18);}
        footer svg{height:30px;vertical-align:middle;margin-right:1rem;}
        """
    )


def brand_header(title: str) -> str:
    """Official dark lockup (which already carries 'by TensorPlay')."""
    return (
        "<header class='tp-nav'>"
        f"<a class='tp-brand' href='https://tensorplay.org' aria-label='MegaRL by TensorPlay'>"
        f"{lockup_svg()}</a>"
        f"<span class='tp-tag'>megaRL / {title}</span>"
        f"<span class='tp-tag' style='margin-left:auto'>{TAGLINE}</span>"
        "</header>"
    )


def brand_footer() -> str:
    return (
        "<footer>"
        f"{symbol_svg()}"
        f"<span>{ENDORSEMENT} &nbsp;·&nbsp; {TAGLINE}</span>"
        "</footer>"
    )
