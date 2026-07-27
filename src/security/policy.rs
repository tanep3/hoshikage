use std::net::IpAddr;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthPolicy {
    LoopbackOpen,
    BearerRequired,
}

impl AuthPolicy {
    pub fn for_bind_host(host: &str) -> Self {
        if host.eq_ignore_ascii_case("localhost") {
            return Self::LoopbackOpen;
        }
        match host.parse::<IpAddr>() {
            Ok(address) if address.is_loopback() => Self::LoopbackOpen,
            _ => Self::BearerRequired,
        }
    }

    pub fn requires_bearer(self) -> bool {
        matches!(self, Self::BearerRequired)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loopback_is_open_and_wildcard_or_hostname_requires_bearer() {
        assert_eq!(
            AuthPolicy::for_bind_host("127.0.0.1"),
            AuthPolicy::LoopbackOpen
        );
        assert_eq!(AuthPolicy::for_bind_host("::1"), AuthPolicy::LoopbackOpen);
        assert_eq!(
            AuthPolicy::for_bind_host("localhost"),
            AuthPolicy::LoopbackOpen
        );
        assert_eq!(
            AuthPolicy::for_bind_host("0.0.0.0"),
            AuthPolicy::BearerRequired
        );
        assert_eq!(
            AuthPolicy::for_bind_host("192.168.1.10"),
            AuthPolicy::BearerRequired
        );
        assert_eq!(
            AuthPolicy::for_bind_host("hoshikage.local"),
            AuthPolicy::BearerRequired
        );
    }
}
