from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import parse_qs, urlencode, urlparse

import requests


AUTH_URL = "https://account.box.com/api/oauth2/authorize"
TOKEN_URL = "https://api.box.com/oauth2/token"
CURRENT_USER_URL = "https://api.box.com/2.0/users/me"


@dataclass(frozen=True)
class BoxAuthResult:
    access_token: str
    refresh_token: str
    expires_in: int | None
    user_id: str
    user_name: str
    user_login: str


def build_authorize_url(client_id: str, redirect_uri: str) -> str:
    client_id = client_id.strip()
    redirect_uri = redirect_uri.strip()

    if not client_id:
        raise ValueError("Client ID is required.")
    if not redirect_uri:
        raise ValueError("Redirect URI is required.")

    query = urlencode(
        {
            "response_type": "code",
            "client_id": client_id,
            "redirect_uri": redirect_uri,
        }
    )
    return f"{AUTH_URL}?{query}"


def extract_authorization_code(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("Paste the redirected localhost URL or authorization code.")

    parsed = urlparse(value)
    if parsed.query:
        params = parse_qs(parsed.query)

        error = params.get("error", [None])[0]
        if error:
            description = params.get("error_description", [""])[0]
            message = f"Box authorization failed: {error}"
            if description:
                message += f" — {description}"
            raise ValueError(message)

        code = params.get("code", [None])[0]
        if code:
            return code

    # Allow the user to paste only the code, matching the old CLI script.
    if "://" not in value and "code=" not in value:
        return value

    raise ValueError(
        "No authorization code was found. Paste the full redirected URL "
        "from the browser address bar."
    )


def exchange_code_for_tokens(
    *,
    client_id: str,
    client_secret: str,
    redirect_uri: str,
    code: str,
) -> dict:
    try:
        response = requests.post(
            TOKEN_URL,
            data={
                "grant_type": "authorization_code",
                "code": code,
                "client_id": client_id.strip(),
                "client_secret": client_secret,
                "redirect_uri": redirect_uri.strip(),
            },
            timeout=30,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Could not contact Box: {exc}") from exc

    if response.status_code != 200:
        try:
            payload = response.json()
            description = (
                payload.get("error_description")
                or payload.get("error")
                or response.text
            )
        except ValueError:
            description = response.text

        raise RuntimeError(
            f"Box token exchange failed ({response.status_code}): {description}"
        )

    payload = response.json()

    if not payload.get("access_token"):
        raise RuntimeError("Box did not return an access token.")

    return payload


def get_current_user(access_token: str) -> dict:
    try:
        response = requests.get(
            CURRENT_USER_URL,
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=30,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Could not verify the Box account: {exc}") from exc

    if response.status_code != 200:
        raise RuntimeError(
            f"Box account verification failed ({response.status_code}): "
            f"{response.text}"
        )

    return response.json()


def complete_oauth(
    *,
    client_id: str,
    client_secret: str,
    redirect_uri: str,
    redirected_url_or_code: str,
) -> BoxAuthResult:
    if not client_secret:
        raise ValueError("Client Secret is required.")

    code = extract_authorization_code(redirected_url_or_code)

    token_data = exchange_code_for_tokens(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
        code=code,
    )

    access_token = token_data["access_token"]
    refresh_token = token_data.get("refresh_token", "")
    user = get_current_user(access_token)

    return BoxAuthResult(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=token_data.get("expires_in"),
        user_id=str(user.get("id", "")),
        user_name=str(user.get("name") or user.get("login") or "Box User"),
        user_login=str(user.get("login") or ""),
    )
