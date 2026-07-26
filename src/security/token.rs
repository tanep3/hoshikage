use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use rand::rngs::OsRng;
use rand::RngCore;
use thiserror::Error;
use zeroize::Zeroize;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum TokenError {
    #[error("token name must match [a-z0-9][a-z0-9._-]{{0,63}}")]
    InvalidName,
    #[error("Bearer token has an invalid format")]
    InvalidFormat,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TokenName(String);

impl TokenName {
    pub fn new(value: impl Into<String>) -> Result<Self, TokenError> {
        let value = value.into();
        let valid = !value.is_empty()
            && value.len() <= 64
            && value.bytes().enumerate().all(|(index, byte)| {
                byte.is_ascii_lowercase()
                    || byte.is_ascii_digit()
                    || (index > 0 && matches!(byte, b'.' | b'_' | b'-'))
            });
        if !valid {
            return Err(TokenError::InvalidName);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

pub struct SecretToken {
    value: String,
    public_id: String,
}

impl SecretToken {
    pub fn generate() -> Self {
        let mut public_id = [0_u8; 16];
        let mut secret = [0_u8; 32];
        OsRng.fill_bytes(&mut public_id);
        OsRng.fill_bytes(&mut secret);
        let public_id = URL_SAFE_NO_PAD.encode(public_id);
        let value = format!("hsk_{}_{}", public_id, URL_SAFE_NO_PAD.encode(secret));
        secret.zeroize();
        Self { value, public_id }
    }

    pub fn parse(value: impl Into<String>) -> Result<Self, TokenError> {
        let value = value.into();
        const PREFIX: &str = "hsk_";
        const PUBLIC_ID_LENGTH: usize = 22;
        const SECRET_LENGTH: usize = 43;
        const SEPARATOR_INDEX: usize = PREFIX.len() + PUBLIC_ID_LENGTH;
        const TOKEN_LENGTH: usize = SEPARATOR_INDEX + 1 + SECRET_LENGTH;

        if value.len() != TOKEN_LENGTH
            || !value.starts_with(PREFIX)
            || value.as_bytes().get(SEPARATOR_INDEX) != Some(&b'_')
        {
            return Err(TokenError::InvalidFormat);
        }
        let public_id = &value[PREFIX.len()..SEPARATOR_INDEX];
        let secret = &value[SEPARATOR_INDEX + 1..];
        if URL_SAFE_NO_PAD.decode(public_id).map(|bytes| bytes.len()) != Ok(16)
            || URL_SAFE_NO_PAD.decode(secret).map(|bytes| bytes.len()) != Ok(32)
        {
            return Err(TokenError::InvalidFormat);
        }
        Ok(Self {
            public_id: public_id.to_string(),
            value,
        })
    }

    pub fn public_id(&self) -> &str {
        &self.public_id
    }

    pub fn expose_secret(&self) -> &str {
        &self.value
    }
}

impl Drop for SecretToken {
    fn drop(&mut self) {
        self.value.zeroize();
        self.public_id.zeroize();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_token_has_expected_entropy_shape_and_round_trips() {
        let token = SecretToken::generate();
        assert!(token.expose_secret().starts_with("hsk_"));
        assert_eq!(
            SecretToken::parse(token.expose_secret())
                .unwrap()
                .public_id(),
            token.public_id()
        );
    }

    #[test]
    fn token_names_and_malformed_secrets_are_rejected() {
        assert!(TokenName::new("codex-lan").is_ok());
        assert!(TokenName::new("Codex").is_err());
        assert!(TokenName::new("../codex").is_err());
        assert!(SecretToken::parse("hsk_short_secret").is_err());
    }

    #[test]
    fn token_parser_accepts_base64url_underscores_inside_both_segments() {
        let public_id = URL_SAFE_NO_PAD.encode([u8::MAX; 16]);
        let secret = URL_SAFE_NO_PAD.encode([u8::MAX; 32]);
        let token = SecretToken::parse(format!("hsk_{public_id}_{secret}")).unwrap();

        assert_eq!(token.public_id(), public_id);
    }
}
