// Copyright (c) 2026 Jason Brazeal (dev@jasonbrazeal.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//! Bearer-token middleware guarding every HTTP request.
//!
//! The presented token is compared with the configured one in constant
//! time. Both are hashed first, so the comparison time does not reveal the
//! token's length either. A rejected request gets HTTP 401 with a fixed
//! body, and an accepted request is forwarded without its `Authorization`
//! header, so nothing downstream can log the token.

use std::sync::Arc;

use axum::extract::{Request, State};
use axum::http::header::{AUTHORIZATION, WWW_AUTHENTICATE};
use axum::http::{HeaderValue, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse as _, Response};
use sha2::{Digest as _, Sha256};
use subtle::ConstantTimeEq as _;

use crate::server::config::BearerToken;

/// Authentication scheme expected in the `Authorization` header, matched
/// case-insensitively.
const BEARER_SCHEME: &str = "Bearer";

/// Body of every 401 response. It never echoes what the client sent.
const UNAUTHORIZED_BODY: &str = "missing or invalid bearer token";

/// Digest of the configured token, the only form the middleware keeps.
#[derive(Clone)]
pub(crate) struct ExpectedToken {
    digest: Arc<[u8]>,
}

impl ExpectedToken {
    /// Keep the digest of `token`.
    pub(crate) fn new(token: &BearerToken) -> Self {
        Self {
            digest: Sha256::digest(token.as_bytes()).to_vec().into(),
        }
    }

    /// Whether `presented` is the configured token, compared in constant time.
    fn matches(&self, presented: &[u8]) -> bool {
        Sha256::digest(presented)
            .as_slice()
            .ct_eq(&self.digest)
            .into()
    }
}

/// Why a request was rejected; logged instead of anything the client sent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Rejection {
    Missing,
    Malformed,
    WrongScheme,
    WrongToken,
}

impl Rejection {
    fn as_str(self) -> &'static str {
        match self {
            Self::Missing => "missing Authorization header",
            Self::Malformed => "malformed Authorization header",
            Self::WrongScheme => "Authorization scheme is not Bearer",
            Self::WrongToken => "wrong bearer token",
        }
    }
}

/// Check the request's bearer token against `expected`.
fn check(request: &Request, expected: &ExpectedToken) -> Result<(), Rejection> {
    let mut values = request.headers().get_all(AUTHORIZATION).iter();
    let value = values.next().ok_or(Rejection::Missing)?;
    if values.next().is_some() {
        return Err(Rejection::Malformed);
    }
    let value = value.to_str().map_err(|_| Rejection::Malformed)?;
    let (scheme, credentials) = value.split_once(' ').ok_or(Rejection::Malformed)?;
    if !scheme.eq_ignore_ascii_case(BEARER_SCHEME) {
        return Err(Rejection::WrongScheme);
    }
    let credentials = credentials.trim_start_matches(' ');
    if credentials.is_empty() {
        return Err(Rejection::Malformed);
    }
    if expected.matches(credentials.as_bytes()) {
        Ok(())
    } else {
        Err(Rejection::WrongToken)
    }
}

/// Axum middleware: forward requests carrying the expected bearer token,
/// stripped of their `Authorization` header, and answer 401 otherwise.
pub(crate) async fn require_bearer(
    State(expected): State<ExpectedToken>,
    mut request: Request,
    next: Next,
) -> Response {
    match check(&request, &expected) {
        Ok(()) => {
            request.headers_mut().remove(AUTHORIZATION);
            next.run(request).await
        }
        Err(rejection) => {
            tracing::warn!(
                reason = rejection.as_str(),
                method = %request.method(),
                path = request.uri().path(),
                "rejected unauthenticated request"
            );
            unauthorized()
        }
    }
}

/// The response to every rejected request.
fn unauthorized() -> Response {
    let mut response = (StatusCode::UNAUTHORIZED, UNAUTHORIZED_BODY).into_response();
    response
        .headers_mut()
        .insert(WWW_AUTHENTICATE, HeaderValue::from_static(BEARER_SCHEME));
    response
}

#[cfg(test)]
mod tests {
    use axum::Router;
    use axum::body::Body;
    use axum::http::HeaderMap;
    use axum::middleware::from_fn_with_state;
    use axum::routing::get;
    use http_body_util::BodyExt as _;
    use tower::ServiceExt as _;

    use super::*;

    const SECRET: &str = "s3cret-Token_value";

    /// Answer whether the request reached the handler with an `Authorization` header.
    async fn echo(headers: HeaderMap) -> &'static str {
        if headers.contains_key(AUTHORIZATION) {
            "leaked"
        } else {
            "stripped"
        }
    }

    fn app() -> Router {
        let token = BearerToken::new(SECRET).unwrap_or_else(|e| panic!("token: {e}"));
        Router::new()
            .route("/", get(echo))
            .layer(from_fn_with_state(
                ExpectedToken::new(&token),
                require_bearer,
            ))
    }

    async fn call(authorization: &[&[u8]]) -> (StatusCode, HeaderMap, String) {
        let mut request = Request::builder().uri("/");
        for value in authorization {
            let value = HeaderValue::from_bytes(value).unwrap_or_else(|e| panic!("header: {e}"));
            request = request.header(AUTHORIZATION, value);
        }
        let request = request
            .body(Body::empty())
            .unwrap_or_else(|e| panic!("request: {e}"));
        let response = app()
            .oneshot(request)
            .await
            .unwrap_or_else(|e| panic!("call: {e}"));
        let status = response.status();
        let headers = response.headers().clone();
        let body = response
            .into_body()
            .collect()
            .await
            .unwrap_or_else(|e| panic!("body: {e}"))
            .to_bytes();
        (status, headers, String::from_utf8_lossy(&body).into_owned())
    }

    async fn assert_rejected(authorization: &[&[u8]]) {
        let (status, headers, body) = call(authorization).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED, "{authorization:?}");
        assert_eq!(
            headers.get(WWW_AUTHENTICATE),
            Some(&HeaderValue::from_static("Bearer"))
        );
        assert_eq!(body, UNAUTHORIZED_BODY);
        assert!(!body.contains(SECRET));
    }

    #[tokio::test]
    async fn missing_token_is_rejected() {
        assert_rejected(&[]).await;
    }

    #[tokio::test]
    async fn malformed_tokens_are_rejected() {
        let duplicated = format!("Bearer {SECRET}");
        for value in [
            b"Bearer".as_slice(),
            b"Bearer ",
            b"Bearer    ",
            SECRET.as_bytes(),
            b"Bearer \xff\xfe",
            b"",
        ] {
            assert_rejected(&[value]).await;
        }
        assert_rejected(&[duplicated.as_bytes(), duplicated.as_bytes()]).await;
    }

    #[tokio::test]
    async fn wrong_scheme_is_rejected() {
        for value in [format!("Basic {SECRET}"), format!("Token {SECRET}")] {
            assert_rejected(&[value.as_bytes()]).await;
        }
    }

    #[tokio::test]
    async fn wrong_tokens_are_rejected() {
        let prefix = &SECRET[..SECRET.len() - 1];
        let upper = SECRET.to_uppercase();
        for token in [prefix, &format!("{SECRET}x"), &upper, "other"] {
            assert_rejected(&[format!("Bearer {token}").as_bytes()]).await;
        }
    }

    #[tokio::test]
    async fn correct_token_passes_without_its_header() {
        for value in [format!("Bearer {SECRET}"), format!("bearer  {SECRET}")] {
            let (status, _, body) = call(&[value.as_bytes()]).await;
            assert_eq!(status, StatusCode::OK, "{value}");
            assert_eq!(body, "stripped");
        }
    }
}
