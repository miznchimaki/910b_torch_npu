#!/usr/bin/env bash
# 此脚本由 upload_910b_to_github_fixed.sh 自动生成，用于恢复 GitHub 中的切片文件。
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
restore_one() {
    local target="$1" expected="$2" tmp actual
    shift 2
    tmp="$(mktemp "${target}.restoring.XXXXXX")" || return 1
    if ! cat -- "$@" > "$tmp"; then rm -f -- "$tmp"; return 1; fi
    actual="$(stat -c %s -- "$tmp")"
    if [[ "$actual" != "$expected" ]]; then
        echo "[ERROR] 文件大小检查失败: $target (expected=$expected actual=$actual)" >&2
        rm -f -- "$tmp"
        return 1
    fi
    mv -f -- "$tmp" "$target"
    echo "[OK] 已恢复: $target"
}
restore_one lib/libtorch_npu.so 95209345 lib/libtorch_npu.so.part000 lib/libtorch_npu.so.part001
echo "所有大文件还原完成！"
