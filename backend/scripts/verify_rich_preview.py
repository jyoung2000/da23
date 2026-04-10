#!/usr/bin/env python
"""Verify rich preview machinery for a job. Usage:

    python -m backend.scripts.verify_rich_preview <job_id>

Checks:
  1. Thumbnail endpoint returns 200 with correct headers
  2. Crawler user-agent on /analysis/{job_id} returns OG tags
  3. Dedicated share route returns OG tags
  4. Regular browser request passes through to SPA
"""

import subprocess
import sys
import os


def _header(text):
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m backend.scripts.verify_rich_preview <job_id>")
        print()
        print("Verifies that rich link previews (OG tags + thumbnails) are")
        print("working correctly for the given job.")
        sys.exit(1)

    job_id = sys.argv[1]
    base = os.environ.get("PUBLIC_BASE_URL", "http://localhost:8000").rstrip("/")

    print(f"Verifying rich preview for job: {job_id}")
    print(f"Base URL: {base}")

    # 1. Thumbnail endpoint
    _header("1. Thumbnail endpoint")
    subprocess.run([
        "curl", "-sI", f"{base}/thumbnails/{job_id}.jpg",
    ])

    # 2. Crawler-injected analysis page
    _header("2. Crawler-injected analysis page (Slackbot UA)")
    result = subprocess.run([
        "curl", "-s",
        "-A", "Slackbot-LinkExpanding 1.0 (+https://api.slack.com/robots)",
        f"{base}/analysis/{job_id}",
    ], capture_output=True, text=True, timeout=10)
    print(result.stdout[:2000] if result.stdout else "(empty response)")
    if "og:image" in (result.stdout or ""):
        print("\n  [PASS] og:image tag found")
    else:
        print("\n  [FAIL] og:image tag NOT found")

    # 3. Dedicated share route
    _header("3. Dedicated share route")
    result = subprocess.run([
        "curl", "-s", f"{base}/share/analysis/{job_id}",
    ], capture_output=True, text=True, timeout=10)
    print(result.stdout[:2000] if result.stdout else "(empty response)")
    if "og:title" in (result.stdout or ""):
        print("\n  [PASS] og:title tag found")
    else:
        print("\n  [FAIL] og:title tag NOT found")

    # 4. Real browser pass-through
    _header("4. Browser request (should be SPA, not stub)")
    result = subprocess.run([
        "curl", "-sI",
        "-A", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        f"{base}/analysis/{job_id}",
    ], capture_output=True, text=True, timeout=10)
    print(result.stdout[:500] if result.stdout else "(empty response)")

    # Summary
    _header("Manual verification links")
    print(f"  Facebook debugger:")
    print(f"    https://developers.facebook.com/tools/debug/?q={base}/share/analysis/{job_id}")
    print(f"  Twitter validator:")
    print(f"    https://cards-dev.twitter.com/validator")
    print(f"  Slack: paste this into a channel:")
    print(f"    {base}/share/analysis/{job_id}")
    print()


if __name__ == "__main__":
    main()
