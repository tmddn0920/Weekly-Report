# Head 파라미터 감소 전략 분석

## 파라미터 감소 옵션 3가지

---

## 옵션 1: 채널 수 감소

Head의 모든 채널을 일정 비율로 감소

```yaml
# 원본
head:
  - [-1, 2, C3k2, [512, False]]  # 512 채널
  - [-1, 2, C3k2, [256, False]]  # 256 채널
  - [-1, 2, C3k2, [512, False]]  # 512 채널
  - [-1, 2, C3k2, [1024, True]]  # 1024 채널

# 50% 감소
head:
  - [-1, 2, C3k2, [256, False]]  # 512×0.5
  - [-1, 2, C3k2, [128, False]]  # 256×0.5
  - [-1, 2, C3k2, [256, False]]  # 512×0.5
  - [-1, 2, C3k2, [512, True]]   # 1024×0.5

# 75% 감소
head:
  - [-1, 2, C3k2, [384, False]]  # 512×0.75
  - [-1, 2, C3k2, [192, False]]  # 256×0.75
  - [-1, 2, C3k2, [384, False]]  # 512×0.75
  - [-1, 2, C3k2, [768, True]]   # 1024×0.75
```

### 파라미터 감소 효과
- **50% 감소**: 채널당 파라미터 `(0.5)² = 0.25` → **약 75% 감소**
- **75% 감소**: 채널당 파라미터 `(0.75)² ≈ 0.56` → **약 44% 감소**

---

## 옵션 2: C3K2 반복 횟수 1회로 줄이기

### 적용 방법
C3K2의 `n` 값을 2에서 1로 감소

```yaml
# 원본 (n=2)
head:
  - [-1, 2, C3k2, [512, False]]  # n=2

# 감소 (n=1)
head:
  - [-1, 1, C3k2, [512, False]]  # n=1
```

### 파라미터 감소 효과
- **n=2 → n=1**: 각 C3K2 블록 내부 Bottleneck이 1개로 감소
- **감소율**: 약 **30-40%** (C3K2 부분만)

---

## 옵션 3: DWConv 사용

### 적용 방법
C3K2 내부의 Bottleneck을 DWConv 기반으로 교체

```python
# (scripts/custom_modules_dw.py 참고)
class C3k2_DW(nn.Module):
    """DWConv 기반 경량화 C3K2"""
    def __init__(self, c1, c2, n=1, e=0.5, shortcut=False):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = DWConv(c1, 2 * self.c, 1)  # DWConv 사용
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        
        # DWConv 기반 경량화 Bottleneck
        self.m = nn.ModuleList(
            nn.Sequential(
                DWConv(self.c, self.c, 3),  # Depth-wise
                Conv(self.c, self.c, 1)      # Point-wise
            ) for _ in range(n)
        )
```

### 파라미터 감소 효과
- **DWConv 효과**: 채널당 파라미터를 약 **1/c로 감소** (c는 채널 수)
- **추가 감소율**: 약 **10-15%** (표준 Conv 대비)
