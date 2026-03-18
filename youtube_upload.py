"""YouTube upload module — OAuth2 + YouTube Data API v3.

Setup:
    1. Create a project at https://console.cloud.google.com
    2. Enable "YouTube Data API v3"
    3. Create OAuth 2.0 Client ID (type: Web application)
    4. Add redirect URI: http://localhost:5000/api/youtube/callback
    5. Download JSON → save as client_secrets.json in this directory
"""

import json
from pathlib import Path

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

CLIENT_SECRETS_FILE = Path(__file__).parent / "client_secrets.json"
TOKEN_FILE = Path(__file__).parent / "youtube_token.json"
SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]


def get_flow(redirect_uri: str) -> Flow:
    """Create an OAuth2 flow from client_secrets.json."""
    return Flow.from_client_secrets_file(
        str(CLIENT_SECRETS_FILE),
        scopes=SCOPES,
        redirect_uri=redirect_uri,
    )


def save_credentials(credentials: Credentials):
    """Persist OAuth credentials to disk."""
    token_data = {
        "token": credentials.token,
        "refresh_token": credentials.refresh_token,
        "token_uri": credentials.token_uri,
        "client_id": credentials.client_id,
        "client_secret": credentials.client_secret,
        "scopes": list(credentials.scopes or SCOPES),
    }
    TOKEN_FILE.write_text(json.dumps(token_data, indent=2), encoding="utf-8")


def load_credentials() -> Credentials | None:
    """Load and refresh stored credentials. Returns None if unavailable."""
    if not TOKEN_FILE.exists():
        return None
    try:
        data = json.loads(TOKEN_FILE.read_text(encoding="utf-8"))
        creds = Credentials(
            token=data["token"],
            refresh_token=data.get("refresh_token"),
            token_uri=data.get("token_uri", "https://oauth2.googleapis.com/token"),
            client_id=data.get("client_id"),
            client_secret=data.get("client_secret"),
            scopes=data.get("scopes", SCOPES),
        )
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            save_credentials(creds)
        return creds if creds.valid else None
    except Exception as e:
        print(f"YouTube token load failed: {e}")
        return None


def is_authenticated() -> bool:
    """Check if valid YouTube credentials exist."""
    return load_credentials() is not None


def upload_video(
    video_path: str,
    title: str,
    description: str = "",
    tags: list[str] = None,
    privacy: str = "private",
    on_progress: callable = None,
) -> dict:
    """Upload a video to YouTube with resumable upload and progress.

    Args:
        video_path:   Path to the video file.
        title:        Video title.
        description:  Video description.
        tags:         List of tags.
        privacy:      "private", "unlisted", or "public".
        on_progress:  Optional callback(percent: int).

    Returns:
        YouTube API response dict (contains "id" = video ID).
    """
    creds = load_credentials()
    if creds is None:
        raise RuntimeError("Not authenticated with YouTube. Please authorize first.")

    youtube = build("youtube", "v3", credentials=creds)

    body = {
        "snippet": {
            "title": title,
            "description": description,
            "tags": tags or [],
            "categoryId": "24",  # Entertainment
        },
        "status": {
            "privacyStatus": privacy,
            "selfDeclaredMadeForKids": False,
        },
    }

    media = MediaFileUpload(
        video_path,
        chunksize=10 * 1024 * 1024,  # 10MB chunks — larger = fewer requests, more reliable
        resumable=True,
        mimetype="video/mp4",
    )

    insert_request = youtube.videos().insert(
        part="snippet,status",
        body=body,
        media_body=media,
    )

    response = None
    retries = 0
    max_retries = 5
    while response is None:
        try:
            status, response = insert_request.next_chunk()
            if status and on_progress:
                on_progress(int(status.progress() * 100))
            retries = 0  # reset on success
        except Exception as e:
            retries += 1
            if retries > max_retries:
                raise RuntimeError(f"Upload failed after {max_retries} retries: {e}")
            import time
            wait = min(2 ** retries, 60)
            print(f"  Upload chunk error (retry {retries}/{max_retries}): {e}")
            print(f"    Waiting {wait}s…")
            if on_progress:
                on_progress(-1)  # signal retry
            time.sleep(wait)

    if on_progress:
        on_progress(100)

    video_id = response["id"]
    print(f"  Upload complete: https://www.youtube.com/watch?v={video_id}")
    return response
