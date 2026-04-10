"""Tests for the crawler-aware OG injection middleware."""

import pytest

from backend.middleware.og_injection import (
    is_crawler,
    render_og_html,
    ANALYSIS_PATH_RE,
    SEO_PATH_RE,
)


class TestIsCrawler:
    def test_slackbot(self):
        assert is_crawler("Slackbot-LinkExpanding 1.0 (+https://api.slack.com/robots)") is True

    def test_twitterbot(self):
        assert is_crawler("Twitterbot/1.0") is True

    def test_facebookbot(self):
        assert is_crawler("facebookexternalhit/1.1 (+http://www.facebook.com/externalhit_uatext.php)") is True

    def test_discordbot(self):
        assert is_crawler("Mozilla/5.0 (compatible; Discordbot/2.0; +https://discordapp.com)") is True

    def test_linkedinbot(self):
        assert is_crawler("LinkedInBot/1.0 (compatible; Mozilla/5.0)") is True

    def test_whatsapp(self):
        assert is_crawler("WhatsApp/2.23.20.0 A") is True

    def test_regular_browser_not_crawler(self):
        assert is_crawler("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36") is False

    def test_empty_string(self):
        assert is_crawler("") is False

    def test_none(self):
        assert is_crawler(None) is False

    def test_case_insensitive(self):
        assert is_crawler("SLACKBOT-LINKEXPANDING 1.0") is True


class TestRenderOgHtml:
    def test_contains_all_required_og_tags(self):
        html = render_og_html(
            title="Test Video",
            description="A test description",
            image_url="https://example.com/thumb.jpg",
            canonical_url="https://example.com/analysis/abc",
        )
        assert 'og:title' in html
        assert 'og:description' in html
        assert 'og:image' in html
        assert 'og:url' in html
        assert 'og:type' in html
        assert 'og:site_name' in html

    def test_contains_twitter_card_tags(self):
        html = render_og_html(
            title="Test",
            description="Desc",
            image_url="https://example.com/thumb.jpg",
            canonical_url="https://example.com/analysis/abc",
        )
        assert 'twitter:card' in html
        assert 'summary_large_image' in html
        assert 'twitter:title' in html
        assert 'twitter:description' in html
        assert 'twitter:image' in html

    def test_absolute_image_url(self):
        html = render_og_html(
            title="Test",
            description="Desc",
            image_url="https://example.com/thumbnails/abc.jpg",
            canonical_url="https://example.com/analysis/abc",
        )
        assert 'content="https://example.com/thumbnails/abc.jpg"' in html

    def test_meta_refresh_redirect(self):
        html = render_og_html(
            title="Test",
            description="Desc",
            image_url="https://example.com/thumb.jpg",
            canonical_url="https://example.com/analysis/abc",
        )
        assert 'http-equiv="refresh"' in html
        assert 'url=https://example.com/analysis/abc' in html

    def test_xss_protection_in_title(self):
        html = render_og_html(
            title='<script>alert("xss")</script>',
            description="Safe",
            image_url="https://example.com/thumb.jpg",
            canonical_url="https://example.com/analysis/abc",
        )
        assert '<script>' not in html
        assert '&lt;script&gt;' in html

    def test_xss_protection_in_description(self):
        html = render_og_html(
            title="Safe",
            description='<img src=x onerror="alert(1)">',
            image_url="https://example.com/thumb.jpg",
            canonical_url="https://example.com/analysis/abc",
        )
        # The raw <img> tag should be escaped so it doesn't render as HTML
        assert '<img src=x' not in html
        assert '&lt;img' in html

    def test_video_tags_included_when_video_url_set(self):
        html = render_og_html(
            title="Test",
            description="Desc",
            image_url="https://example.com/thumb.jpg",
            canonical_url="https://example.com/seo/abc",
            video_url="https://example.com/video.mp4",
        )
        assert 'og:video' in html
        assert 'twitter:player' in html

    def test_no_video_tags_when_none(self):
        html = render_og_html(
            title="Test",
            description="Desc",
            image_url="https://example.com/thumb.jpg",
            canonical_url="https://example.com/analysis/abc",
        )
        assert 'og:video' not in html

    def test_image_dimensions_specified(self):
        html = render_og_html(
            title="Test",
            description="Desc",
            image_url="https://example.com/thumb.jpg",
            canonical_url="https://example.com/analysis/abc",
        )
        assert 'og:image:width' in html
        assert 'og:image:height' in html
        assert '1200' in html
        assert '630' in html


class TestPathRegex:
    def test_analysis_path_matches(self):
        m = ANALYSIS_PATH_RE.match("/analysis/abc123")
        assert m is not None
        assert m.group("job_id") == "abc123"

    def test_analysis_path_with_trailing_slash(self):
        m = ANALYSIS_PATH_RE.match("/analysis/abc123/")
        assert m is not None

    def test_analysis_path_rejects_nested(self):
        m = ANALYSIS_PATH_RE.match("/analysis/abc/def")
        assert m is None

    def test_seo_path_matches(self):
        m = SEO_PATH_RE.match("/seo/job123/clip456")
        assert m is not None
        assert m.group("job_id") == "job123"
        assert m.group("clip_id") == "clip456"
