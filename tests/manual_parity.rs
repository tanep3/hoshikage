fn numbered_headings(document: &str) -> Vec<String> {
    document
        .lines()
        .filter_map(|line| {
            let heading = line.strip_prefix('#')?.trim_start_matches('#').trim_start();
            let number = heading.split_whitespace().next()?;
            number
                .chars()
                .next()
                .is_some_and(|value| value.is_ascii_digit())
                .then(|| number.trim_end_matches('.').to_string())
        })
        .collect()
}

fn code_blocks(document: &str) -> Vec<String> {
    let mut blocks = Vec::new();
    let mut current = Vec::new();
    let mut inside = false;
    for line in document.lines() {
        if line.starts_with("```") {
            if inside {
                blocks.push(current.join("\n"));
                current.clear();
            }
            inside = !inside;
        } else if inside {
            current.push(line.to_string());
        }
    }
    assert!(!inside, "manual contains an unclosed code block");
    blocks
}

#[test]
fn japanese_and_english_manuals_have_matching_structure_and_commands() {
    let japanese = include_str!("../docs/user-manual.md");
    let english = include_str!("../docs/user-manual.en.md");

    assert_eq!(numbered_headings(japanese), numbered_headings(english));
    assert_eq!(code_blocks(japanese), code_blocks(english));
    assert!(japanese.contains("[English](user-manual.en.md)"));
    assert!(english.contains("[日本語](user-manual.md)"));
}

#[test]
fn manuals_include_end_user_codex_setup_contract() {
    for manual in [
        include_str!("../docs/user-manual.md"),
        include_str!("../docs/user-manual.en.md"),
    ] {
        assert!(manual.contains(r"%USERPROFILE%\.codex\config.toml"));
        assert!(manual.contains(r"%USERPROFILE%\.codex\hoshikage.config.toml"));
        assert!(manual.contains("~/.config/hoshikage"));
        assert!(manual.contains("~/Library/Application Support/hoshikage"));
        assert!(manual.contains(r"%APPDATA%\hoshikage"));
        assert!(manual.contains(".codex/config.toml"));
        assert!(manual.contains("HOSHIKAGE_API_KEY"));
        assert!(manual.contains("hoshikage token list"));
        assert!(manual.contains("<unavailable: rotate required>"));
        assert!(manual.contains("codex exec --profile hoshikage"));
        assert!(manual.contains("192.168.1.50"));
    }
}
