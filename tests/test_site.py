"""Tests for the static presentation website (presentation/build_site.py)."""

import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "presentation"))

import build_site  # noqa: E402
from views.stories import STORIES  # noqa: E402  (app/ is on sys.path through build_site)


class Links(HTMLParser):
    """Collects local links, scripts, stylesheets and iframes of a page."""

    def __init__(self):
        super().__init__()
        self.targets = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        for name in ("href", "src"):
            value = attrs.get(name)
            if value and not value.startswith(("http://", "https://", "data:", "#", "mailto:")):
                self.targets.append(value)


@pytest.fixture(scope="module")
def site(tmp_path_factory) -> Path:
    directory = tmp_path_factory.mktemp("site")
    build_site.build(directory, progress=False)
    return directory


def test_all_pages_are_written(site):
    assert (site / "index.html").exists()
    assert (site / ".nojekyll").exists()
    for key, story in STORIES.items():
        assert (site / f"{key}.html").exists()
        assert (site / f"rennen-{key}.html").exists() == bool(story.race)


def test_every_local_link_points_to_an_existing_file(site):
    for page in site.glob("*.html"):
        parser = Links()
        parser.feed(page.read_text(encoding="utf-8"))
        for target in parser.targets:
            assert (site / target.split("#")[0]).exists(), f"{page.name} links to missing {target}"


def test_no_external_resources(site):
    """The site must work offline: no scripts or stylesheets from other servers."""
    for page in site.glob("*.html"):
        text = page.read_text(encoding="utf-8")
        assert not re.search(r'<(script|link)[^>]+(src|href)="https?://', text), page.name


@pytest.mark.parametrize("key", list(STORIES))
def test_story_data_is_complete(site, key):
    text = (site / f"{key}.html").read_text(encoding="utf-8")
    data = json.loads(re.search(r'<script type="application/json" id="story-data">(.*?)</script>', text, re.S).group(1))
    assert len(data["points"]) > 1
    assert any(c["recommended"] for c in data["counts"])
    assert len(data["counts"]) >= 2, "the recommendation is compared with at least one alternative"
    what_if = data["what_if"]
    assert str(what_if["default_q"]) in what_if["decisions"]
    names = {c["name"] for c in what_if["curves"]}
    assert all(d["recommended"] in names for d in what_if["decisions"].values()), "every recommendation has its own curve"
    assert STORIES[key].lesson in text.replace("&quot;", '"')
