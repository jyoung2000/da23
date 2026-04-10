#!/usr/bin/env python
"""Verify rich preview machinery for a job. Usage:

    python -m backend.scripts.verify_rich_preview <job_id>

Checks:
  1. Thumbnail endpoint returns image/jpeg
  2. Crawler-injected analysis page returns HTML with og:image
  3. Dedicated share route returns HTML with OG tags
  4. Browser request passes through to SPA (no stub)
"""

import subprocess
import sys
import os


def main():
    if len(sys.argv) < 2:
        print("usage: verify_rich_preview <job_id>")
        sys.exit(1)

    job_id = sys.argv[1]
    base = os.environ.get("PUBLIC_BASE_URL", "http://localhost:8000").rstrip("/")

    print(f"Verifying rich preview for job {job_id} on {base}")
    print("-" * 60)

    # 1. Thumbnail endpoint
    print("\n1. Thumbnail endpoint:")
    subprocess.run(["curl", "-sI", f"{base}/thumbnails/{job_id}.jpg"])

    # 2. Crawler-injected analysis page
    print("\n2. Crawler-injected analysis page (Slackbot UA):")
    result = subprocess.run([
        "curl", "-s",
        "-A", "Slackbot-LinkExpanding 1.0 (+https://api.slack.com/robots)",
        f"{base}/analysis/{job_id}",
    ], capture_output=True, text=True)
    print(result.stdout[:500] if result.stdout else "(empty)")
    if "og:image" in (result.stdout or ""):
        print("  [PASS] og:image tag found")
    else:
        print("  [WARN] og:image tag NOT found")

    # 3. Share route
    print("\n3. Dedicated share route:")
    result = subprocess.run([
        "curl", "-s", f"{base}/share/analysis/{job_id}",
    ], capture_output=True, text=True)
    print(result.stdout[:500] if result.stdout else "(empty)")
    if "og:title" in (result.stdout or ""):
        print("  [PASS] og:title tag found")
    else:
        print("  [WARN] og:title tag NOT found")

    # 4. Browser request
    print("\n4. Browser request to /analysis (should be SPA, not stub):")
    result = subprocess.run([
        "curl", "-sI",
        "-A", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        f"{base}/analysis/{job_id}",
    ], capture_output=True, text=True)
    print(result.stdout[:300] if result.stdout else "(empty)")

    print("\n" + "-" * 60)
    print("Manual verification links:")
    print(f"  Facebook debugger:  https://developers.facebook.com/tools/debug/?q={base}/share/analysis/{job_id}")
    print(f"  Twitter validator:  https://cards-dev.twitter.com/validator")
    print(f"  Slack: paste {base}/share/analysis/{job_id} into a channel")


if __name__ == "__main__":
    main()
