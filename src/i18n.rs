#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Language {
    En,
    Ja,
}

impl Language {
    pub fn resolve(explicit: Option<Self>, configured: Option<&str>, locale: Option<&str>) -> Self {
        explicit
            .or_else(|| configured.and_then(Self::parse))
            .or_else(|| locale.and_then(Self::from_locale))
            .unwrap_or(Self::En)
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "en" | "english" => Some(Self::En),
            "ja" | "jp" | "japanese" => Some(Self::Ja),
            _ => None,
        }
    }

    fn from_locale(locale: &str) -> Option<Self> {
        let language = locale
            .split(['.', '_', '-'])
            .next()
            .unwrap_or(locale)
            .to_ascii_lowercase();
        match language.as_str() {
            "ja" => Some(Self::Ja),
            "en" => Some(Self::En),
            _ => None,
        }
    }

    pub fn select<'a>(self, en: &'a str, ja: &'a str) -> &'a str {
        match self {
            Self::En => en,
            Self::Ja => ja,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalizedText {
    en: String,
    ja: String,
}

impl LocalizedText {
    pub fn new(en: impl Into<String>, ja: impl Into<String>) -> Self {
        Self {
            en: en.into(),
            ja: ja.into(),
        }
    }

    pub fn get(&self, language: Language) -> &str {
        language.select(&self.en, &self.ja)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn language_resolution_uses_explicit_then_config_then_locale_then_english() {
        assert_eq!(
            Language::resolve(Some(Language::Ja), Some("en"), Some("en_US.UTF-8")),
            Language::Ja
        );
        assert_eq!(
            Language::resolve(None, Some("ja"), Some("en_US.UTF-8")),
            Language::Ja
        );
        assert_eq!(
            Language::resolve(None, None, Some("ja_JP.UTF-8")),
            Language::Ja
        );
        assert_eq!(
            Language::resolve(None, None, Some("fr_FR.UTF-8")),
            Language::En
        );
        assert_eq!(Language::resolve(None, None, None), Language::En);
    }

    #[test]
    fn machine_values_are_not_produced_by_localizer() {
        assert_eq!(Language::En.select("Ready", "準備完了"), "Ready");
        assert_eq!(Language::Ja.select("Ready", "準備完了"), "準備完了");
    }

    #[test]
    fn localized_text_keeps_both_human_languages_out_of_machine_values() {
        let text = LocalizedText::new("Model is ready", "モデルは準備完了です");
        assert_eq!(text.get(Language::En), "Model is ready");
        assert_eq!(text.get(Language::Ja), "モデルは準備完了です");
    }
}
