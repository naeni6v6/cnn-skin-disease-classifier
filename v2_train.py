"""
S-CURT 피부질환 7-class 분류 학습 파이프라인 (PyTorch / EfficientNetV2-S)
────────────────────────────────────────────────────────────────────────
입력: prepare_dataset.py 결과물
      C:\\Users\\user\\Desktop\\S-CURT_clean\\
          0 normal\\000000.jpg ...
          1 여드름(Acne)\\...
          ...
          manifest.csv        ← 있으면 자동으로 활용 (pseudo_source 층화/누출진단)
          report.txt

핵심 설계 의도
  1) manifest.csv의 pseudo_source와 pHash를 split에 반영
     다른 클래스의 이미지는 삭제하지 않고 모두 유지하므로, 같은 사진이나
     근접중복 사진이 train/val/test에 갈라지지 않도록 pHash 그룹 단위로 분할함.
     특정 촬영 시그니처가 클래스에 몰린 문제는 출처별 정확도로 별도 진단함.
     (--source-holdout을 쓰면 출처 단위로 test를 떼어냄)

  2) 클래스 불균형 → soft class weight + (옵션) WeightedRandomSampler
     완전 balanced 가중치는 소수 클래스 과적합을 유발하므로 beta=0.5로 완화.

  3) 평가 기준은 accuracy가 아니라 macro-F1
     불균형 데이터에서 accuracy는 다수 클래스(습진 33.8%)에 끌려감.

  4) 2단계 파인튜닝: backbone freeze 워밍업 → 전체 해제
     소규모 의료 데이터에서 처음부터 전체를 풀면 사전학습 특징이 망가짐.

실행
    pip install torch torchvision timm scikit-learn pandas numpy tqdm pillow
    python train.py                    # 기본 학습
    python train.py --kfold 5          # K-Fold 교차검증
    python train.py --source-holdout   # 출처 단위 test 분리 (엄격 평가)
    python train.py --export-onnx      # 학습 후 ONNX 변환까지
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
    from PIL import Image
except ImportError as e:
    sys.exit(f"필수 패키지 없음: {e.name}\n설치: pip install numpy pandas pillow")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
except ImportError:
    sys.exit(
        "PyTorch가 현재 Python 환경에 설치되어 있지 않음.\n"
        "가상환경을 활성화한 뒤 torch와 torchvision을 설치할 것."
    )

try:
    from sklearn.metrics import (classification_report, confusion_matrix,
                                 f1_score)
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:
    sys.exit("설치 필요: pip install -U scikit-learn")

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

Image.MAX_IMAGE_PIXELS = None


# ══════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════

DATA_ROOT = Path(r"C:\Users\user\Desktop\S-CURT_clean")
OUT_ROOT = Path(r"C:\Users\user\Desktop\S-CURT_runs")

CLASS_DIRS = [
    "0 normal",
    "1 여드름(Acne)",
    "2 습진(Eczema)",
    "3 벌레물림(Insect_Bites)",
    "4 흑색종_멜라노마(Melanoma)",
    "5 건선(Psoriasis)",
    "6 진균감염(Fungal_Infection)",
]
# 앱/보고서에 쓸 영문 라벨 (폴더명 순서와 1:1 대응)
CLASS_NAMES_EN = [
    "Normal", "Acne", "Eczema", "Insect_Bites",
    "Melanoma", "Psoriasis", "Fungal_Infection",
]

# ── 모델 ───────────────────────────────────────────────
# timm 이름. 설치 안 돼 있으면 torchvision efficientnet_v2_s로 자동 폴백.
MODEL_NAME = "tf_efficientnetv2_s.in21k_ft_in1k"
IMG_SIZE = 300          # EfficientNetV2-S 학습 해상도
EVAL_IMG_SIZE = 300     # 384로 올리면 소폭 개선되나 VRAM/시간 증가
DROPOUT = 0.3           # 과적합 대응 (원본 프로젝트 이슈)
DROP_PATH = 0.2         # stochastic depth

# ── 학습 ───────────────────────────────────────────────
# RTX 3060 12GB 기준. VRAM 6GB 노트북 3060이면 BATCH_SIZE=16으로 낮출 것.
BATCH_SIZE = 32
NUM_WORKERS = 4         # Windows는 4 이상에서 오히려 느려지는 경우가 많음
EPOCHS = 40
FREEZE_EPOCHS = 3       # backbone 동결 워밍업
LR_HEAD = 1e-3          # 분류기 학습률
LR_BACKBONE = 1e-4      # backbone 파인튜닝 학습률
WEIGHT_DECAY = 1e-4     # L2 regularization
WARMUP_EPOCHS = 2
LABEL_SMOOTHING = 0.05
EARLY_STOP_PATIENCE = 8
USE_AMP = True          # mixed precision. 3060이면 켜는 게 무조건 이득
GRAD_CLIP = 1.0

# ── 불균형 처리 ─────────────────────────────────────────
# beta=0: 가중치 없음 / 0.5: 완화된 balanced / 1.0: 완전 balanced
CLASS_WEIGHT_BETA = 0.5
USE_BALANCED_SAMPLER = False   # class weight와 동시 사용은 과보정이라 비권장

# ── split ──────────────────────────────────────────────
VAL_RATIO = 0.15
TEST_RATIO = 0.15
DUP_HASH_THRESHOLD = 5   # 근접중복 이미지는 삭제하지 않고 같은 split에 묶음
SEED = 42

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


DETERMINISTIC = False   # --deterministic 로 켜짐. 켜면 재현성↑ 속도↓


def set_seed(seed=SEED, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


def _worker_init(worker_id: int):
    s = SEED + worker_id
    np.random.seed(s)
    random.seed(s)


# ══════════════════════════════════════════════════════════
# 1. 데이터 인덱싱
# ══════════════════════════════════════════════════════════

def build_dataframe(root: Path) -> pd.DataFrame:
    """
    manifest.csv가 있으면 pseudo_source, vcluster, pHash 메타를 사용함.
    없으면 폴더를 직접 스캔하되, 중복 그룹 분할은 비활성화됨.
    반환 컬럼: path, label, class_dir, source, vcluster, phash
    """
    manifest = root / "manifest.csv"
    if manifest.exists():
        df = pd.read_csv(manifest, encoding="utf-8-sig")
        df = df.rename(columns={
            "new_path": "path",
            "class": "class_dir",
            "pseudo_source": "source",
        })
        keep = ["path", "class_dir", "source", "vcluster", "phash"]
        for c in keep:
            if c not in df.columns:
                df[c] = "-"
        df = df[keep].copy()

        # manifest가 다른 위치에서 만들어졌더라도 현재 root 기준으로 경로 보정
        missing = ~df["path"].astype(str).map(lambda p: Path(p).exists())
        if missing.any():
            df.loc[missing, "path"] = df.loc[missing].apply(
                lambda r: str(root / str(r["class_dir"]) / Path(str(r["path"])).name),
                axis=1,
            )

        exists = df["path"].astype(str).map(lambda p: Path(p).exists())
        n_missing = int((~exists).sum())
        if n_missing:
            print(f"  ⚠ manifest 경로 누락 {n_missing}장 제외")
        df = df[exists].drop_duplicates("path").reset_index(drop=True)
        print(f"  manifest.csv 사용 — {len(df)}장 (출처/pHash 메타 포함)")
    else:
        rows = []
        for cdir in CLASS_DIRS:
            d = root / cdir
            if not d.exists():
                print(f"  ! 폴더 없음: {cdir}")
                continue
            for p in sorted(d.rglob("*")):
                if p.suffix.lower() in EXTS:
                    rows.append({
                        "path": str(p),
                        "class_dir": cdir,
                        "source": "-",
                        "vcluster": "-",
                        "phash": "-",
                    })
        df = pd.DataFrame(rows)
        print(f"  폴더 스캔 — {len(df)}장")
        print("  ⚠ manifest.csv 없음: pHash 중복 그룹 분할과 출처 진단 비활성")

    if df.empty:
        return df

    cls_to_idx = {c: i for i, c in enumerate(CLASS_DIRS)}
    df = df[df["class_dir"].isin(cls_to_idx)].copy()
    df["label"] = df["class_dir"].map(cls_to_idx).astype(int)
    df["source"] = df["source"].fillna("-").astype(str)
    df["vcluster"] = df["vcluster"].fillna("-").astype(str)
    df["phash"] = df["phash"].fillna("-").astype(str).str.lower()
    return df.reset_index(drop=True)


def assign_duplicate_groups(
    df: pd.DataFrame,
    threshold: int = DUP_HASH_THRESHOLD,
) -> pd.DataFrame:
    """
    이미지는 삭제하지 않음.
    pHash 해밍거리 <= threshold인 이미지를 한 그룹으로 묶어
    같은 그룹이 train/val/test에 갈라지지 않게 함.
    """
    out = df.copy()
    n = len(out)
    if n == 0:
        out["dup_group"] = []
        return out

    valid = out["phash"].str.fullmatch(r"[0-9a-f]{16}", na=False).to_numpy()
    valid_idx = np.flatnonzero(valid)

    if len(valid_idx) == 0:
        out["dup_group"] = [f"unique_{i:07d}" for i in range(n)]
        print("  ⚠ 유효한 pHash 없음: 각 이미지를 독립 그룹으로 처리")
        return out

    bits = {int(i): int(out.iloc[i]["phash"], 16) for i in valid_idx}
    parent = {int(i): int(i) for i in valid_idx}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # 64bit 전체를 threshold+1개 밴드로 나눠 후보를 좁힘
    n_bands = threshold + 1
    buckets = {}
    for i in valid_idx:
        i = int(i)
        v = bits[i]
        for b in range(n_bands):
            start = (64 * b) // n_bands
            end = (64 * (b + 1)) // n_bands
            width = end - start
            mask = (1 << width) - 1
            key = (b, (v >> start) & mask)
            buckets.setdefault(key, []).append(i)

    checked = set()
    for idxs in buckets.values():
        if len(idxs) < 2 or len(idxs) > 700:
            continue
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                i, j = idxs[a], idxs[b]
                pair = (i, j) if i < j else (j, i)
                if pair in checked:
                    continue
                checked.add(pair)
                if (bits[i] ^ bits[j]).bit_count() <= threshold:
                    union(i, j)

    groups = []
    for i in range(n):
        if valid[i]:
            groups.append(f"phash_{find(i):07d}")
        else:
            groups.append(f"unique_{i:07d}")
    out["dup_group"] = groups

    sizes = out["dup_group"].value_counts()
    multi = sizes[sizes > 1]
    print(
        f"  pHash 그룹 {len(sizes)}개 | "
        f"중복 그룹 {len(multi)}개 / 포함 이미지 {int(multi.sum())}장"
    )
    return out


def _group_split(
    df: pd.DataFrame,
    holdout_ratio: float,
    seed: int,
):
    """StratifiedGroupKFold로 목표 비율에 가까운 한 fold를 분리."""
    n_splits = max(2, int(round(1.0 / holdout_ratio)))
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    keep_idx, hold_idx = next(splitter.split(
        df,
        y=df["label"],
        groups=df["dup_group"],
    ))
    return (
        df.iloc[keep_idx].reset_index(drop=True),
        df.iloc[hold_idx].reset_index(drop=True),
    )


def split_data(df: pd.DataFrame, source_holdout=False, seed=SEED):
    """
    기본:
      pHash 근접중복 그룹을 보존한 채 클래스 층화 70/15/15에 가깝게 분할.
      중복 이미지를 삭제하지 않으면서 train/test 누출을 막음.

    --source-holdout:
      클래스별 특정 pseudo_source를 test로 분리한 뒤, 해당 test 이미지와
      같은 pHash 그룹도 전부 test에 포함. 나머지는 그룹 단위 train/val 분할.
    """
    if "dup_group" not in df.columns:
        df = assign_duplicate_groups(df)

    if not source_holdout:
        train_val, test = _group_split(df, TEST_RATIO, seed)
        val_ratio_in_rest = VAL_RATIO / max(1e-6, 1.0 - len(test) / len(df))
        train, val = _group_split(train_val, val_ratio_in_rest, seed + 1)
        return train, val, test

    rng = np.random.default_rng(seed)
    test_idx = []

    for lab, g in df.groupby("label"):
        srcs = [s for s in g["source"].unique() if s != "-"]
        if len(srcs) < 2:
            continue

        want = max(1, int(len(g) * TEST_RATIO))
        sizes = sorted(
            ((s, int((g["source"] == s).sum())) for s in srcs),
            key=lambda x: x[1],
        )
        chosen, acc = [], 0
        for src, count in sizes:
            if acc >= want:
                break
            if count > len(g) * 0.5:
                continue
            if chosen and acc + count > want * 1.4:
                continue
            chosen.append(src)
            acc += count

        if chosen:
            test_idx.extend(g.index[g["source"].isin(chosen)].tolist())

    selected = set(test_idx)

    # 출처 분리가 안 된 클래스는 임시로 랜덤 보충
    for lab, g in df.groupby("label"):
        current = sum(1 for i in selected if int(df.loc[i, "label"]) == lab)
        need = int(len(g) * TEST_RATIO) - current
        candidates = [i for i in g.index if i not in selected]
        if need > 0 and candidates:
            picked = rng.choice(
                candidates,
                size=min(need, max(1, len(candidates) - 1)),
                replace=False,
            )
            selected.update(int(i) for i in np.atleast_1d(picked))

    # 선택된 이미지와 동일/근접중복인 그룹 전체를 test로 이동.
    # 큰 중복 그룹이 딸려오면 test 가 목표의 2~3배로 부풀 수 있으므로 확인.
    selected_groups = set(df.loc[sorted(selected), "dup_group"])
    test = df[df["dup_group"].isin(selected_groups)].reset_index(drop=True)
    rest = df[~df["dup_group"].isin(selected_groups)].reset_index(drop=True)

    actual = len(test) / max(1, len(df))
    print(f"  [source-holdout] test 비율 {actual*100:.1f}% "
          f"(목표 {TEST_RATIO*100:.0f}%)")
    if actual > TEST_RATIO * 2:
        print(f"  ⚠ test 가 목표의 2배를 넘음 — 중복 그룹 확장 때문.")
        print(f"     train 이 부족해질 수 있으니 클래스별 분포를 확인할 것.")
    for lab in sorted(df["label"].unique()):
        n_all = int((df["label"] == lab).sum())
        n_te = int((test["label"] == lab).sum()) if len(test) else 0
        if n_all and n_te / n_all > 0.45:
            print(f"  ⚠ '{CLASS_NAMES_EN[lab]}' 의 {n_te/n_all*100:.0f}% 가 "
                  f"test 로 빠짐 (출처 편중 클래스)")

    val_ratio_in_rest = VAL_RATIO / max(1e-6, 1.0 - len(test) / len(df))
    train, val = _group_split(rest, val_ratio_in_rest, seed + 1)
    return train, val, test


# ══════════════════════════════════════════════════════════
# 2. Dataset / 증강
# ══════════════════════════════════════════════════════════

try:
    from torchvision import transforms as T
except ImportError:
    sys.exit("설치 필요: pip install torchvision")


def build_transforms(train: bool):
    if train:
        return T.Compose([
            T.RandomResizedCrop(IMG_SIZE, scale=(0.70, 1.0), ratio=(0.8, 1.25)),
            T.RandomHorizontalFlip(),
            # VerticalFlip 제거: 멜라노마 ABCD의 비대칭성 단서를 훼손함.
            # 방향 다양성은 아래 RandomRotation 이 대신 제공.
            T.RandomApply([T.RandomRotation(30)], p=0.5),
            # 조명 편향만 완화. 홍반 강도(saturation)와 색조(hue)는
            # 습진/건선/진균 감별의 핵심 단서라 최소한만 흔듦.
            T.ColorJitter(brightness=0.20, contrast=0.20,
                          saturation=0.10, hue=0.01),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            # scale 상한 축소: 소형 병변(벌레물림, 초기 멜라노마)이
            # 통째로 지워지는 것을 방지
            T.RandomErasing(p=0.20, scale=(0.02, 0.08)),
        ])
    return T.Compose([
        T.Resize(int(EVAL_IMG_SIZE * 1.14)),
        T.CenterCrop(EVAL_IMG_SIZE),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class SkinDataset(Dataset):
    def __init__(self, df: pd.DataFrame, train: bool):
        self.paths = df["path"].tolist()
        self.labels = df["label"].tolist()
        self.tf = build_transforms(train)
        self._broken = 0

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            with Image.open(self.paths[i]) as im:
                img = im.convert("RGB")
        except Exception as e:
            # 손상 파일은 학습을 중단시키지 않되, 조용히 넘기지 않고 로그를 남김.
            # 검정(0,0,0) 대신 중간 회색을 쓰는 이유: 검정이 특정 클래스에
            # 몰리면 모델이 '검은 이미지 = 그 클래스'로 학습해버림.
            self._broken += 1
            if self._broken <= 20:
                print(f"    ! 손상 이미지({type(e).__name__}): {self.paths[i]}")
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (128, 128, 128))
        return self.tf(img), self.labels[i]


def verify_images(df: pd.DataFrame, workers: int = 8) -> pd.DataFrame:
    """
    학습 전에 손상 이미지를 미리 걸러냄.
    __getitem__ 의 회색 대체는 최후의 안전망일 뿐, 사전 제거가 정석.
    """
    from concurrent.futures import ThreadPoolExecutor

    def _ok(p):
        try:
            with Image.open(p) as im:
                im.verify()
            return True
        except Exception:
            return False

    paths = df["path"].tolist()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        flags = list(tqdm(ex.map(_ok, paths), total=len(paths),
                          desc="  이미지 검증", ncols=80, leave=False))
    flags = np.array(flags, dtype=bool)
    n_bad = int((~flags).sum())
    if n_bad:
        print(f"  ⚠ 손상 이미지 {n_bad}장 제외")
        for p in np.array(paths)[~flags][:10]:
            print(f"     - {p}")
    else:
        print("  이미지 검증 통과 (손상 0장)")
    return df[flags].reset_index(drop=True)


def make_loaders(train_df, val_df, test_df):
    train_ds = SkinDataset(train_df, train=True)
    sampler, shuffle = None, True
    if USE_BALANCED_SAMPLER:
        cnt = Counter(train_df["label"])
        w = np.array([1.0 / cnt[l] for l in train_df["label"]], dtype=np.float64)
        sampler = WeightedRandomSampler(torch.from_numpy(w), len(w), True)
        shuffle = False

    common = dict(
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=NUM_WORKERS > 0,
        worker_init_fn=_worker_init if NUM_WORKERS > 0 else None,
    )
    if NUM_WORKERS > 0:
        common["prefetch_factor"] = 2
    return (
        DataLoader(train_ds, BATCH_SIZE, shuffle=shuffle, sampler=sampler,
                   drop_last=True,
                   generator=torch.Generator().manual_seed(SEED),
                   **common),
        DataLoader(SkinDataset(val_df, False), BATCH_SIZE * 2, shuffle=False,
                   **common),
        DataLoader(SkinDataset(test_df, False), BATCH_SIZE * 2, shuffle=False,
                   **common),
    )


# ══════════════════════════════════════════════════════════
# 3. 모델
# ══════════════════════════════════════════════════════════

def _detect_head_keys(model) -> tuple:
    """
    분류기 파라미터 접두사를 실제 모델에서 탐지함.
    timm 버전/아키텍처에 따라 classifier / head.fc / head 등으로 달라지므로
    하드코딩하면 head_params 가 빈 리스트가 되어 AdamW 가 죽음.
    """
    names = [n for n, _ in model.named_parameters()]
    for cand in ("classifier.", "head.fc.", "head.", "fc."):
        if any(n.startswith(cand) for n in names):
            return (cand.rstrip("."),)
    # 최후 수단: 마지막 파라미터가 속한 최상위 모듈명을 사용
    fallback = names[-1].split(".")[0]
    print(f"  ⚠ 표준 head 이름을 찾지 못해 '{fallback}' 로 대체함")
    return (fallback,)


def build_model(n_classes: int):
    try:
        import timm
    except ImportError:
        timm = None

    if timm is not None:
        try:
            m = timm.create_model(MODEL_NAME, pretrained=True,
                                  num_classes=n_classes,
                                  drop_rate=DROPOUT, drop_path_rate=DROP_PATH)
            head_keys = _detect_head_keys(m)
            print(f"  timm — {MODEL_NAME} | head={head_keys[0]}")
            return m, head_keys
        except Exception as e:
            # 가중치 다운로드 실패/네트워크 문제도 여기로 옴 → 이유를 명시
            print(f"  timm 모델 생성 실패({type(e).__name__}: {e})")

    from torchvision.models import (EfficientNet_V2_S_Weights,
                                    efficientnet_v2_s)
    print("  → torchvision efficientnet_v2_s 로 폴백")
    m = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    in_f = m.classifier[-1].in_features
    m.classifier = nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(in_f, n_classes))
    head_keys = _detect_head_keys(m)
    print(f"  head={head_keys[0]}")
    return m, head_keys


def set_backbone_frozen(model, head_keys, frozen: bool):
    for name, p in model.named_parameters():
        p.requires_grad = (not frozen) or name.startswith(head_keys)


def class_weights(train_df, n_classes, beta=CLASS_WEIGHT_BETA):
    cnt = np.array([max(1, int((train_df["label"] == i).sum()))
                    for i in range(n_classes)], dtype=np.float64)
    w = (cnt.sum() / (n_classes * cnt)) ** beta
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def cosine_lr(step, total_steps, warmup_steps):
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))


def make_sched_fn(freeze_steps: int, warmup_steps: int):
    """
    원본은 warmup(2ep)과 freeze(3ep)가 겹쳐서, head 만 학습하는 구간에서
    head LR 이 warmup 으로 눌려 있었음 → head 가 덜 익은 채 backbone 이 풀림.

    수정: freeze 구간은 head 를 full LR 로 학습시키고,
          unfreeze 시점부터 warmup + cosine 을 새로 시작함.
    """
    def _fn(step, total_steps):
        if step < freeze_steps:
            return 1.0                       # freeze 구간: head full LR
        return cosine_lr(step - freeze_steps,
                         max(1, total_steps - freeze_steps),
                         warmup_steps)
    return _fn


# ══════════════════════════════════════════════════════════
# 4. 학습 / 평가 루프
# ══════════════════════════════════════════════════════════

try:
    from torch.amp import GradScaler, autocast
    def _scaler(enabled): return GradScaler("cuda", enabled=enabled)
    def _autocast(enabled): return autocast("cuda", enabled=enabled)
except ImportError:                                     # torch < 2.0
    from torch.cuda.amp import GradScaler, autocast
    def _scaler(enabled): return GradScaler(enabled=enabled)
    def _autocast(enabled): return autocast(enabled=enabled)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device,
                    sched_fn, global_step, total_steps, base_lrs,
                    frozen: bool = False):
    model.train()
    loss_sum, seen, correct = 0.0, 0, 0
    for x, y in tqdm(loader, desc="    train", ncols=80, leave=False):
        f = sched_fn(global_step, total_steps)
        for gi, g in enumerate(optimizer.param_groups):
            # gi==0 은 backbone. freeze 중에는 LR 을 0 으로 눌러
            # weight decay 로 인한 미세 드리프트까지 차단.
            g["lr"] = 0.0 if (frozen and gi == 0) else base_lrs[gi] * f

        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if device.type == "cuda":
            x = x.contiguous(memory_format=torch.channels_last)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(USE_AMP and device.type == "cuda"):
            out = model(x)
            loss = criterion(out, y)
        scaler.scale(loss).backward()
        if GRAD_CLIP:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        loss_sum += loss.item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        seen += y.size(0)
        global_step += 1
    return loss_sum / max(1, seen), correct / max(1, seen), global_step


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, ys, ps, probs = 0.0, [], [], []
    for x, y in tqdm(loader, desc="    eval ", ncols=80, leave=False):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if device.type == "cuda":
            x = x.contiguous(memory_format=torch.channels_last)
        with _autocast(USE_AMP and device.type == "cuda"):
            out = model(x)
            loss = criterion(out, y)
        loss_sum += loss.item() * y.size(0)
        probs.append(F.softmax(out.float(), 1).cpu().numpy())
        ps.append(out.argmax(1).cpu().numpy())
        ys.append(y.cpu().numpy())
    ys, ps = np.concatenate(ys), np.concatenate(ps)
    return {
        "loss": loss_sum / max(1, len(ys)),
        "acc": float((ys == ps).mean()),
        "macro_f1": float(f1_score(ys, ps, average="macro", zero_division=0)),
        "y_true": ys, "y_pred": ps, "probs": np.concatenate(probs),
    }


def run_training(train_df, val_df, test_df, device, out_dir: Path, tag=""):
    n_classes = len(CLASS_DIRS)
    train_loader, val_loader, test_loader = make_loaders(train_df, val_df, test_df)

    model, head_keys = build_model(n_classes)
    model.to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    w = class_weights(train_df, n_classes).to(device)
    print(f"  class weight: {[round(v,2) for v in w.tolist()]}")
    criterion = nn.CrossEntropyLoss(weight=w, label_smoothing=LABEL_SMOOTHING)

    head_params = [p for n, p in model.named_parameters() if n.startswith(head_keys)]
    bb_params = [p for n, p in model.named_parameters() if not n.startswith(head_keys)]
    if not head_params:
        sample = [n for n, _ in model.named_parameters()][-5:]
        raise RuntimeError(
            f"head_keys={head_keys} 로 분류기 파라미터를 찾지 못함.\n"
            f"실제 파라미터 예시: {sample}"
        )
    if not bb_params:
        raise RuntimeError("backbone 파라미터가 비어 있음. head_keys 탐지 오류.")
    print(f"  param split — head {len(head_params)}개 / backbone {len(bb_params)}개")

    # param_groups[0]=backbone, [1]=head 순서 고정 (아래 freeze 로직이 의존함)
    optimizer = torch.optim.AdamW(
        [{"params": bb_params, "lr": LR_BACKBONE},
         {"params": head_params, "lr": LR_HEAD}],
        weight_decay=WEIGHT_DECAY)
    base_lrs = [g["lr"] for g in optimizer.param_groups]

    steps_per_epoch = max(1, len(train_loader))
    total_steps = EPOCHS * steps_per_epoch
    freeze_steps = FREEZE_EPOCHS * steps_per_epoch
    warmup_steps = WARMUP_EPOCHS * steps_per_epoch
    sched_fn = make_sched_fn(freeze_steps, warmup_steps)

    scaler = _scaler(USE_AMP and device.type == "cuda")
    best_f1, best_epoch, bad, gstep, history = -1.0, -1, 0, 0, []
    ckpt = out_dir / f"best{tag}.pt"

    for ep in range(1, EPOCHS + 1):
        frozen = ep <= FREEZE_EPOCHS
        set_backbone_frozen(model, head_keys, frozen)
        t0 = time.time()
        tr_loss, tr_acc, gstep = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            sched_fn, gstep, total_steps, base_lrs, frozen=frozen)
        va = evaluate(model, val_loader, criterion, device)

        cur_lr = optimizer.param_groups[1]["lr"]
        print(f"  [{ep:02d}/{EPOCHS}] {'(freeze)' if frozen else '        '} "
              f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va['loss']:.4f} acc {va['acc']:.4f} "
              f"macroF1 {va['macro_f1']:.4f} | lr {cur_lr:.2e} "
              f"| {time.time()-t0:.0f}s")
        history.append({"epoch": ep, "frozen": frozen,
                        "lr_head": cur_lr,
                        "lr_backbone": optimizer.param_groups[0]["lr"],
                        "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_loss": va["loss"], "val_acc": va["acc"],
                        "val_macro_f1": va["macro_f1"]})

        if va["macro_f1"] > best_f1:
            best_f1, best_epoch, bad = va["macro_f1"], ep, 0
            # 원자적 저장: 저장 도중 중단돼도 기존 best 가 손상되지 않음
            tmp = ckpt.with_suffix(".tmp")
            torch.save({"model": model.state_dict(), "epoch": ep,
                        "macro_f1": best_f1, "classes": CLASS_DIRS,
                        "classes_en": CLASS_NAMES_EN, "model_name": MODEL_NAME,
                        "head_keys": list(head_keys),
                        "img_size": IMG_SIZE}, tmp)
            tmp.replace(ckpt)
        else:
            # freeze 워밍업 동안의 미개선은 early stopping에 포함하지 않음
            if not frozen:
                bad += 1
                if bad >= EARLY_STOP_PATIENCE:
                    print(f"  early stop (patience {EARLY_STOP_PATIENCE})")
                    break

    pd.DataFrame(history).to_csv(out_dir / f"history{tag}.csv", index=False)
    print(f"  best epoch {best_epoch}  val macro-F1 {best_f1:.4f}")

    if not ckpt.exists():
        print("  ⚠ best 체크포인트가 없음(전 epoch 미개선). "
              "마지막 epoch 가중치로 test 를 진행함.")
    else:
        try:
            saved = torch.load(ckpt, map_location=device, weights_only=False)
        except TypeError:  # 구버전 PyTorch 호환
            saved = torch.load(ckpt, map_location=device)
        model.load_state_dict(saved["model"])
    te = evaluate(model, test_loader, criterion, device)
    print(f"\n  [TEST] acc {te['acc']:.4f}  macro-F1 {te['macro_f1']:.4f}")
    return model, te, test_df, best_f1


# ══════════════════════════════════════════════════════════
# 5. 리포트
# ══════════════════════════════════════════════════════════

def write_report(out_dir: Path, res, test_df, tag=""):
    y, p = res["y_true"], res["y_pred"]
    rep = classification_report(y, p, labels=list(range(len(CLASS_DIRS))),
                                target_names=CLASS_NAMES_EN, digits=4,
                                zero_division=0)
    cm = confusion_matrix(y, p, labels=list(range(len(CLASS_DIRS))))

    lines = [f"S-CURT test 결과{tag}", "",
             f"accuracy  {res['acc']:.4f}",
             f"macro-F1  {res['macro_f1']:.4f}", "", rep, "",
             "[혼동행렬]  행=정답 열=예측",
             "            " + " ".join(f"{n[:6]:>7s}" for n in CLASS_NAMES_EN)]
    for i, row in enumerate(cm):
        lines.append(f"{CLASS_NAMES_EN[i][:11]:<11s} " +
                     " ".join(f"{v:7d}" for v in row))

    # 출처별 정확도 분해 — prepare 단계에서 경고한 '촬영방식 학습' 검증용
    # y/p 는 DataLoader 순서(=test_df 행 순서)의 위치 기반 배열이므로
    # 길이가 어긋나면 조용히 잘못된 통계가 나옴 → 명시적으로 검증.
    if "source" in test_df.columns and test_df["source"].nunique() > 1:
        if len(test_df) != len(y):
            lines += ["", f"[출처별 정확도] 생략 — 길이 불일치 "
                          f"(test_df {len(test_df)} vs 예측 {len(y)})"]
        else:
            lines += ["", "[출처별 정확도] 편차가 크면 병변이 아니라 촬영방식을 학습한 것"]
            tmp = test_df.reset_index(drop=True).copy()
            tmp["_ok"] = np.asarray(y == p)
            for src, g in tmp.groupby("source"):
                if len(g) < 20:
                    continue
                lines.append(f"  {str(src):<38s} {g['_ok'].mean():.4f}  (n={len(g)})")

    # confidence 분포 — 앱의 '전문의 상담 권고' 임계값 근거
    conf = res["probs"].max(1)
    lines += ["", "[confidence 분포]",
              f"  전체 평균 {conf.mean():.4f}",
              f"  정답 평균 {conf[y == p].mean():.4f}"]
    if (y != p).any():
        lines.append(f"  오답 평균 {conf[y != p].mean():.4f}")
    lines.append("  임계값별 커버리지/정확도:")
    for th in (0.5, 0.6, 0.7, 0.8, 0.9):
        m = conf >= th
        if m.sum():
            lines.append(f"    conf>={th:.1f}  커버 {m.mean()*100:5.1f}%  "
                         f"정확도 {(y[m] == p[m]).mean():.4f}")

    # 클래스별 confidence — 전체 평균만 보면 다수 클래스에 가려짐.
    # 서비스 임계값은 클래스별로 달라야 하므로 분해해서 기록.
    lines += ["", "[클래스별 예측 confidence / recall]"]
    for i, cn in enumerate(CLASS_NAMES_EN):
        m_true = (y == i)
        if not m_true.any():
            continue
        rec = float((p[m_true] == i).mean())
        m_pred = (p == i)
        c_mean = float(conf[m_pred].mean()) if m_pred.any() else float("nan")
        lines.append(f"  {cn:<18s} recall {rec:.4f}  "
                     f"예측시 평균conf {c_mean:.4f}  (n_true={int(m_true.sum())})")

    txt = "\n".join(l for l in lines if l)
    (out_dir / f"test_report{tag}.txt").write_text(txt, encoding="utf-8")
    print("\n" + rep)
    np.save(out_dir / f"confusion{tag}.npy", cm)


def export_onnx(model, out_dir: Path, device):
    """
    channels_last + AMP 상태 그대로 export 하면 실패/경고가 잦음.
    CPU + contiguous + float32 로 되돌린 뒤 export 하고, 끝나면 원복.
    """
    path = out_dir / "model.onnx"
    try:
        m = (model.to("cpu")
                  .to(memory_format=torch.contiguous_format)
                  .float()
                  .eval())
        dummy = torch.randn(1, 3, EVAL_IMG_SIZE, EVAL_IMG_SIZE)
        with torch.no_grad():
            torch.onnx.export(
                m, dummy, str(path),
                input_names=["input"], output_names=["logits"],
                opset_version=17,
                dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
            )
        print(f"  ONNX 저장: {path}")
    except ModuleNotFoundError as e:
        print(f"  ONNX 변환 패키지 없음({e.name}). "
              f"설치: pip install onnx onnxscript")
    except Exception as e:
        print(f"  ONNX 변환 실패({type(e).__name__}): {e}")
    finally:
        model.to(device)
        if device.type == "cuda":
            model.to(memory_format=torch.channels_last)


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

def main():
    global EPOCHS, BATCH_SIZE, IMG_SIZE, EVAL_IMG_SIZE, NUM_WORKERS

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=str(DATA_ROOT))
    ap.add_argument("--out", type=str, default=str(OUT_ROOT))
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--img-size", type=int, default=IMG_SIZE)
    ap.add_argument("--workers", type=int, default=NUM_WORKERS)
    ap.add_argument("--eval-img-size", type=int, default=0,
                    help="0이면 --img-size 와 동일. 384 등으로 따로 지정 가능")
    ap.add_argument("--kfold", type=int, default=0, help="0이면 단일 split")
    ap.add_argument("--source-holdout", action="store_true",
                    help="출처 단위로 test 분리 (엄격 평가)")
    ap.add_argument("--export-onnx", action="store_true")
    ap.add_argument("--deterministic", action="store_true",
                    help="완전 재현성 우선 (속도 10~20%% 저하)")
    ap.add_argument("--skip-verify", action="store_true",
                    help="손상 이미지 사전 검증 건너뜀")
    args = ap.parse_args()

    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    IMG_SIZE = args.img_size
    # eval 해상도를 train 과 분리 (원본은 항상 덮어써서 384 실험이 불가능했음)
    EVAL_IMG_SIZE = args.eval_img_size if args.eval_img_size > 0 else args.img_size
    NUM_WORKERS = max(0, args.workers)

    set_seed(SEED, deterministic=args.deterministic)
    root = Path(args.data)
    if not root.exists():
        sys.exit(f"경로 없음: {root}\nprepare_dataset.py를 먼저 실행할 것.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = not args.deterministic
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"\n[1/4] 환경   device=cuda | {gpu_name} | VRAM {gpu_mem:.1f}GB")
    else:
        print("\n[1/4] 환경   device=cpu")
        print("  ⚠ GPU가 인식되지 않아 학습이 매우 느릴 수 있음.")
        print("  현재 가상환경에 CUDA 지원 torch가 설치됐는지 확인할 것.")

    print(f"\n[2/4] 데이터   {root}")
    df = build_dataframe(root)
    if df.empty:
        sys.exit("이미지를 찾지 못함. 폴더명이 CLASS_DIRS와 일치하는지 확인할 것.")

    cnt = Counter(df["label"])
    missing_classes = [CLASS_DIRS[i] for i in range(len(CLASS_DIRS))
                       if cnt.get(i, 0) == 0]
    if missing_classes:
        sys.exit("비어 있는 클래스 폴더가 있음:\n  " + "\n  ".join(missing_classes))

    for i, c in enumerate(CLASS_DIRS):
        n = cnt.get(i, 0)
        print(f"  {c:<34s}{n:7d}{n/len(df)*100:7.1f}%")
    nz = [v for v in cnt.values() if v]
    print(f"  {'합계':<34s}{len(df):7d}   최대/최소 {max(nz)/min(nz):.1f} : 1")

    if not args.skip_verify:
        df = verify_images(df)
        if df.empty:
            sys.exit("검증 후 남은 이미지가 없음.")

    # pHash 가 비어 있으면 누출 방지 로직 전체가 무력화되므로 반드시 경고
    valid_ph = df["phash"].str.fullmatch(r"[0-9a-f]{16}", na=False).sum()
    ph_ratio = valid_ph / len(df)
    print(f"  유효 pHash {valid_ph}/{len(df)} ({ph_ratio*100:.1f}%)")
    if ph_ratio < 0.5:
        print("  ⚠⚠ 유효 pHash 비율이 50% 미만임.")
        print("     근접중복이 train/test 로 갈라져 성능이 과대평가될 수 있음.")
        print("     prepare_dataset.py 에서 phash 컬럼을 채웠는지 확인할 것.")

    # 데이터는 삭제하지 않고, 근접중복끼리 같은 split에 묶음
    df = assign_duplicate_groups(df)
    train_df, val_df, test_df = split_data(df, args.source_holdout)

    print(f"\n[3/4] split   train {len(train_df)} / val {len(val_df)} / "
          f"test {len(test_df)}"
          f"{'  (출처 홀드아웃)' if args.source_holdout else ''}")
    for name, d in (("train", train_df), ("val", val_df), ("test", test_df)):
        c = Counter(d["label"])
        print(f"  {name:<6s}" + " ".join(
            f"{c.get(i, 0):5d}" for i in range(len(CLASS_DIRS))
        ))

    path_overlap = set(train_df["path"]) & set(test_df["path"])
    group_overlap_tv = set(train_df["dup_group"]) & set(val_df["dup_group"])
    group_overlap_tt = set(train_df["dup_group"]) & set(test_df["dup_group"])
    group_overlap_vt = set(val_df["dup_group"]) & set(test_df["dup_group"])
    print(f"  train∩test 경로 중복: {len(path_overlap)}건")
    print(
        "  split 간 pHash 그룹 중복: "
        f"train-val {len(group_overlap_tv)} / "
        f"train-test {len(group_overlap_tt)} / "
        f"val-test {len(group_overlap_vt)}"
    )
    if path_overlap or group_overlap_tv or group_overlap_tt or group_overlap_vt:
        sys.exit("split 누출이 감지되어 학습을 중단함.")

    for name, d in (("train", train_df), ("val", val_df), ("test", test_df)):
        present = set(d["label"].unique())
        missing = [CLASS_NAMES_EN[i] for i in range(len(CLASS_DIRS))
                   if i not in present]
        if missing:
            sys.exit(f"{name} split에 없는 클래스: {', '.join(missing)}")

    out_dir = Path(args.out) / time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "classes.json").write_text(json.dumps(
        {
            "class_dirs": CLASS_DIRS,
            "class_names_en": CLASS_NAMES_EN,
            "img_size": IMG_SIZE,
            "mean": IMAGENET_MEAN,
            "std": IMAGENET_STD,
            "model_name": MODEL_NAME,
            "duplicate_hash_threshold": DUP_HASH_THRESHOLD,
        },
        ensure_ascii=False,
        indent=2,
    ), encoding="utf-8")

    # ── 실험 기록 (포트폴리오/재현성용) ──────────────────────
    def _ver(mod):
        try:
            return __import__(mod).__version__
        except Exception:
            return "-"

    run_meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": vars(args),
        "hyperparams": {
            "model_name": MODEL_NAME,
            "img_size": IMG_SIZE, "eval_img_size": EVAL_IMG_SIZE,
            "batch_size": BATCH_SIZE, "epochs": EPOCHS,
            "freeze_epochs": FREEZE_EPOCHS, "warmup_epochs": WARMUP_EPOCHS,
            "lr_head": LR_HEAD, "lr_backbone": LR_BACKBONE,
            "weight_decay": WEIGHT_DECAY, "dropout": DROPOUT,
            "drop_path": DROP_PATH, "label_smoothing": LABEL_SMOOTHING,
            "class_weight_beta": CLASS_WEIGHT_BETA,
            "use_balanced_sampler": USE_BALANCED_SAMPLER,
            "grad_clip": GRAD_CLIP, "use_amp": USE_AMP,
            "early_stop_patience": EARLY_STOP_PATIENCE,
            "dup_hash_threshold": DUP_HASH_THRESHOLD, "seed": SEED,
        },
        "env": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "-",
            "numpy": _ver("numpy"), "pandas": _ver("pandas"),
            "sklearn": _ver("sklearn"), "timm": _ver("timm"),
            "torchvision": _ver("torchvision"),
        },
        "split_sizes": {"train": len(train_df), "val": len(val_df),
                        "test": len(test_df)},
        "class_counts": {
            CLASS_NAMES_EN[i]: {
                "train": int((train_df["label"] == i).sum()),
                "val": int((val_df["label"] == i).sum()),
                "test": int((test_df["label"] == i).sum()),
            } for i in range(len(CLASS_DIRS))
        },
    }
    (out_dir / "run_meta.json").write_text(
        json.dumps(run_meta, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")

    # split 자체를 저장해야 나중에 동일 조건으로 재현 가능
    for name, d in (("train", train_df), ("val", val_df), ("test", test_df)):
        d[["path", "label", "class_dir", "source", "dup_group"]].to_csv(
            out_dir / f"split_{name}.csv", index=False, encoding="utf-8-sig")

    print(f"\n[4/4] 학습 시작 → {out_dir}\n")

    if args.kfold and args.kfold > 1:
        pool = pd.concat([train_df, val_df]).reset_index(drop=True)
        skf = StratifiedGroupKFold(
            n_splits=args.kfold,
            shuffle=True,
            random_state=SEED,
        )
        scores = []
        split_iter = skf.split(
            pool,
            y=pool["label"],
            groups=pool["dup_group"],
        )
        for k, (tr, va) in enumerate(split_iter, 1):
            print(f"\n──── Fold {k}/{args.kfold} ────")
            _, res, _, bf1 = run_training(
                pool.iloc[tr].reset_index(drop=True),
                pool.iloc[va].reset_index(drop=True),
                test_df,
                device,
                out_dir,
                tag=f"_fold{k}",
            )
            write_report(out_dir, res, test_df, tag=f"_fold{k}")
            scores.append({
                "fold": k,
                "val_macro_f1": bf1,
                "test_acc": res["acc"],
                "test_macro_f1": res["macro_f1"],
            })
            if device.type == "cuda":
                torch.cuda.empty_cache()

        summary = pd.DataFrame(scores)
        summary.to_csv(out_dir / "kfold_summary.csv", index=False)
        print(
            f"\n[K-Fold 요약] val macro-F1 (일반화 성능 추정) "
            f"{summary['val_macro_f1'].mean():.4f} ± "
            f"{summary['val_macro_f1'].std():.4f}"
        )
        print(
            f"  참고) test macro-F1 평균 "
            f"{summary['test_macro_f1'].mean():.4f} ± "
            f"{summary['test_macro_f1'].std():.4f}"
        )
        print("  ※ test set 은 fold 마다 동일한 데이터를 반복 평가한 값이므로")
        print("     일반화 성능 지표로 보고하면 안 됨(낙관적 편향). "
              "대표 지표는 val 평균을 사용할 것.")
        print(summary.to_string(index=False))
    else:
        model, res, test_df, _ = run_training(
            train_df,
            val_df,
            test_df,
            device,
            out_dir,
        )
        write_report(out_dir, res, test_df)
        if args.export_onnx:
            export_onnx(model, out_dir, device)

    print(f"\n완료. 결과: {out_dir}\n")


if __name__ == "__main__":
    main()
