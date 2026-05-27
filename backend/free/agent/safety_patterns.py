"""危険コマンドパターン定義（Free / Pro 共有）"""

# 危険コマンドパターン（run_command_safe + EventReminderSystem で共有）
DANGEROUS_PATTERNS: list[str] = [
    r"rm\s+(-[rf]+\s+)?/",  # rm -rf /
    r"rm\s+-[rf]*\s+\*",  # rm -rf *
    r"mkfs\.",  # mkfs.ext4 etc.
    r"dd\s+.*of=",  # dd of=
    r":\(\)\{\s*:\|:&\s*\};:",  # fork bomb
    r"chmod\s+-R\s+777\s+/",  # chmod -R 777 /
    r">\s*/dev/sd",  # > /dev/sda
]
