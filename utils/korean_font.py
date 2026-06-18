"""matplotlib 한글 폰트 설정 유틸.

사용 예:
    from utils.korean_font import set_korean_font
    set_korean_font()

Windows / macOS / Linux 공통 지원. 설치된 폰트 중 한글 가능한 걸 자동 선택.
"""
from __future__ import annotations

import platform
import warnings
from typing import Iterable

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt


# 우선순위 순 (OS별)
_CANDIDATES = {
    "Windows": ["Malgun Gothic", "NanumGothic", "Gulim", "Dotum", "Batang"],
    "Darwin":  ["AppleGothic", "NanumGothic", "Apple SD Gothic Neo"],
    "Linux":   ["NanumGothic", "Noto Sans CJK KR", "Noto Sans KR", "UnDotum"],
}


def _available_fonts() -> set[str]:
    """시스템에 설치된 폰트 이름 집합."""
    return {f.name for f in fm.fontManager.ttflist}


def set_korean_font(preferred: Iterable[str] | None = None, verbose: bool = True) -> str | None:
    """matplotlib 기본 폰트를 한글 가능한 폰트로 변경.

    Args:
        preferred: 우선 시도할 폰트명 리스트 (선택)
        verbose:   선택된 폰트 출력 여부

    Returns:
        실제 적용된 폰트명 (실패 시 None)
    """
    os_name = platform.system()
    candidates = list(preferred) if preferred else []
    candidates += _CANDIDATES.get(os_name, [])
    # OS 상관없이 한 번 더 훑기
    candidates += _CANDIDATES["Windows"] + _CANDIDATES["Darwin"] + _CANDIDATES["Linux"]
    # 중복 제거 (순서 유지)
    seen, ordered = set(), []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            ordered.append(c)

    installed = _available_fonts()
    chosen: str | None = None
    for name in ordered:
        if name in installed:
            chosen = name
            break

    if chosen is None:
        warnings.warn(
            "한글 폰트를 찾지 못했습니다. Windows면 Malgun Gothic이 기본 설치돼 있어야 합니다. "
            "Linux면 'sudo apt install fonts-nanum' 후 matplotlib 캐시를 지우세요."
        )
        return None

    matplotlib.rcParams["font.family"] = chosen
    # 음수 기호 깨짐 방지 (-1 이 □ 로 표시되는 현상)
    matplotlib.rcParams["axes.unicode_minus"] = False

    if verbose:
        print(f"[korean_font] '{chosen}' 적용 완료 (axes.unicode_minus=False)")
    return chosen


if __name__ == "__main__":
    # 스크립트로 실행 시 자체 테스트
    font = set_korean_font()
    import numpy as np
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(np.arange(-3, 4), np.arange(-3, 4) ** 2, marker="o")
    ax.set_title("한글 폰트 테스트 — 웨이퍼 결함 분포")
    ax.set_xlabel("음수 기호 확인: -1, -2, -3")
    ax.set_ylabel("값")
    plt.tight_layout()
    plt.savefig("/tmp/korean_font_test.png")
    print(f"테스트 이미지 저장: /tmp/korean_font_test.png (font={font})")
