"""포토다이오드 신호로 용매 증발 완료(건조 종료점)를 판정한다.

용매가 날아가는 동안에는 막 두께와 조성이 바뀌면서 반사/투과광 세기가 계속 변한다
(얇은 막에서는 간섭 때문에 오르내리기도 한다). 다 마르면 신호가 평탄해지므로,
최근 window_s 초 동안 중앙값 필터를 거친 신호의 변동폭이 band_pct 이내이면 완료로 본다.
"""
from collections import deque
from statistics import median


class DryEndpointDetector:
    def __init__(self, window_s: float, band_pct: float, smooth_n: int = 5):
        self.window_s = window_s
        self.band = band_pct / 100
        self._raw: deque[float] = deque(maxlen=smooth_n)
        self._points: deque[tuple[float, float]] = deque()

    def update(self, t: float, value: float) -> None:
        self._raw.append(value)
        self._points.append((t, median(self._raw)))
        # 판정 구간 전체를 덮는 데 필요한 만큼만 남긴다
        while len(self._points) > 1 and t - self._points[1][0] >= self.window_s:
            self._points.popleft()

    def span_s(self) -> float:
        if not self._points:
            return 0.0
        return self._points[-1][0] - self._points[0][0]

    def variation_pct(self) -> float | None:
        if not self._points:
            return None
        values = [v for _, v in self._points]
        mean = sum(values) / len(values)
        if abs(mean) < 1e-9:
            return None
        return (max(values) - min(values)) / abs(mean) * 100

    def is_stable(self) -> bool:
        if self.span_s() < self.window_s:
            return False
        variation = self.variation_pct()
        return variation is not None and variation <= self.band * 100

    def snapshot(self) -> dict:
        return {
            "stable": self.is_stable(),
            "variation_pct": self.variation_pct(),
            "band_pct": self.band * 100,
            "span_s": self.span_s(),
            "window_s": self.window_s,
        }
