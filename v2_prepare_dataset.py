"""
S-CURT 피부질환 데이터셋 전처리 파이프라인
────────────────────────────────────────────────────────────
데이터 출처 (Kaggle 3종 병합):
  - lysaapriani/skin-disease-and-normal-skin-dataset
  - pacificrm/skindiseasedataset
  - ismailpromus/skin-diseases-image-dataset
  ※ shubhamgoel27/dermnet 은 중복으로 제외했으나, 위 3종 자체가
    DermNet 파생일 가능성이 높아 데이터셋 간 중복 검사가 필수임.

주요 기능
  1) pHash / 해상도 / 블러 / JPEG 양자화테이블 계산
  2) 유사 출처(pseudo-source) 자동 추출  = FORMAT_WxH_q해시
     → 병합 과정에서 출처 폴더가 사라졌으므로 이미지 자체에서 복원
  3) 전역 근접중복 탐지 (클래스 내부 = 제거 / 클래스 간 = 라벨 충돌)
  4) 출처 편중 진단  ← 특정 촬영방식이 특정 클래스에만 몰렸는지 검사
     (예: 더모스코피 이미지가 흑색종에만 존재 → 모델이 촬영방식을 학습)
  5) 하위질환 + 유사출처 층화 언더샘플링
  6) 새 폴더로 복사 + manifest.csv / report.txt

하위질환 층화
  클래스 폴더 아래에 하위 폴더가 있으면 그것을 하위질환으로 인식해
  층화 기준에 포함함. 없으면 자동으로 무시하고 진행.
      2 습진(Eczema)\\atopic\\ ...
                    \\seborrheic\\ ...
  습진처럼 여러 질환을 병합한 클래스는 하위 폴더를 만들어두는 편이
  언더샘플링 시 특정 하위질환이 통째로 사라지는 것을 막아줌.

원본은 절대 수정하지 않음. 복사만 함.

실행:
    pip install pillow imagehash numpy tqdm
    python prepare_dataset.py            # 분석만 (dry-run)
    python prepare_dataset.py --apply    # 실제 복사
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import shutil
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

try:
    import imagehash
except ImportError:
    sys.exit("설치 필요: pip install pillow imagehash numpy tqdm")

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

Image.MAX_IMAGE_PIXELS = None


# ══════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════

SRC_ROOT = Path(r"C:\Users\user\Desktop\S-CURT")
DST_ROOT = Path(r"C:\Users\user\Desktop\S-CURT_clean")

CLASS_DIRS = [
    "0 normal",
    "1 여드름(Acne)",
    "2 습진(Eczema)",
    "3 벌레물림(Insect_Bites)",
    "4 흑색종_멜라노마(Melanoma)",
    "5 건선(Psoriasis)",
    "6 진균감염(Fungal_Infection)",
]

# 목표 장수. 미기재 클래스는 전량 유지
TARGET_COUNTS = {
    "2 습진(Eczema)": 4000,
}

# 근접중복 pHash 해밍거리 임계. 0=완전동일, 5≈사실상 같은 사진
# 제거량이 과하면 3으로 낮출 것
HASH_THRESHOLD = 5

# 클래스 간 중복(= 같은 사진이 서로 다른 라벨) 처리 방식
#   "report"     : 리포트만, 데이터는 그대로 (기본. 먼저 이걸로 실태 파악)
#   "drop_all"   : 라벨을 믿을 수 없으므로 양쪽 모두 제거 (가장 보수적)
#   "keep_rarest": 희소 클래스 쪽만 남김 (소수 클래스 데이터 보존)
CROSS_CLASS_POLICY = "report"

# 품질 필터
MIN_SIDE = 200          # 짧은 변 최소 픽셀
BLUR_THRESHOLD = 25.0   # 라플라시안 분산. 0으로 두면 블러 필터 해제

# 유사 출처: 클래스 내 이 비율 미만인 시그니처는 'misc'로 병합
SOURCE_MIN_RATIO = 0.02

# 시각적 군집 개수 (언더샘플링 대상 클래스에만 적용)
# 하위질환 라벨이 없으므로, 색·질감으로 유형을 나눠 다양성을 보존함
N_VISUAL_CLUSTERS = 20

# 출처 편중 경고 기준: 이미지 N장 이상인 출처가 한 클래스에 이 비율 이상 몰리면 경고
LEAK_MIN_IMAGES = 100
LEAK_WARN_RATIO = 0.90

SEED = 42
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
N_WORKERS = 8


# ══════════════════════════════════════════════════════════
# 1. 이미지 분석
# ══════════════════════════════════════════════════════════

def analyze_image(path_str: str):
    """pHash / 해상도 / 블러 / JPEG 양자화테이블 해시 계산."""
    path = Path(path_str)
    try:
        with Image.open(path) as im:
            fmt = (im.format or path.suffix.lstrip(".")).upper()

            # JPEG 양자화 테이블 = 인코더 지문. 같은 데이터셋이면 거의 동일.
            # 주의: exif_transpose()는 새 객체를 반환하며 이 속성을 잃으므로
            #       반드시 변환 '전에' 읽어야 함.
            qsig = "na"
            q = getattr(im, "quantization", None)
            if q:
                flat = ",".join(str(v) for k in sorted(q) for v in list(q[k])[:16])
                qsig = hashlib.md5(flat.encode()).hexdigest()[:6]

            im = ImageOps.exif_transpose(im)
            w, h = im.size
            rgb = im.convert("RGB")
            phash = str(imagehash.phash(rgb, hash_size=8))

            # 클러스터링용 색 특징: 32x32 축소 후 채널별 평균/표준편차
            small = rgb.resize((32, 32), Image.BILINEAR)
            ca = np.asarray(small, dtype=np.float32) / 255.0
            color_feat = np.concatenate([
                ca.reshape(-1, 3).mean(0), ca.reshape(-1, 3).std(0)
            ]).tolist()

            gray = rgb.convert("L")
            gray.thumbnail((256, 256), Image.BILINEAR)
            arr = np.asarray(gray, dtype=np.float64)
            lap = (
                -4 * arr[1:-1, 1:-1]
                + arr[:-2, 1:-1] + arr[2:, 1:-1]
                + arr[1:-1, :-2] + arr[1:-1, 2:]
            )
            blur = float(lap.var()) if lap.size else 0.0

        return {
            "path": str(path),
            "phash": phash,
            "width": w,
            "height": h,
            "min_side": min(w, h),
            "blur": blur,
            "fmt": fmt,
            "qsig": qsig,
            "raw_sig": f"{fmt}_{w}x{h}_q{qsig}",
            "color_feat": color_feat,
        }
    except Exception as e:
        return {"path": str(path), "error": f"{type(e).__name__}: {e}"}


def scan_class(class_root: Path, cname: str):
    files = [p for p in class_root.rglob("*") if p.suffix.lower() in EXTS]
    if not files:
        return []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        recs = list(tqdm(
            ex.map(analyze_image, [str(f) for f in files], chunksize=32),
            total=len(files), desc=f"  {class_root.name}", ncols=80,
        ))
    for r in recs:
        r["cls"] = cname
        if "error" in r:
            continue
        # 하위 폴더가 있으면 하위질환으로 인식
        rel = Path(r["path"]).relative_to(class_root)
        r["subclass"] = rel.parts[0] if len(rel.parts) > 1 else "-"
    return recs


def assign_pseudo_sources(records, min_ratio=SOURCE_MIN_RATIO):
    """raw_sig 빈도 상위만 독립 출처로 인정, 나머지는 'misc'로 병합."""
    if not records:
        return Counter()
    freq = Counter(r["raw_sig"] for r in records)
    cutoff = max(2, int(len(records) * min_ratio))
    major = {sig for sig, n in freq.items() if n >= cutoff}
    for r in records:
        r["source"] = r["raw_sig"] if r["raw_sig"] in major else "misc"
    return Counter(r["source"] for r in records)


def assign_visual_clusters(records, k=N_VISUAL_CLUSTERS, seed=SEED):
    """
    pHash 비트(구조) + 색 통계로 k-means 클러스터링.
    하위질환 라벨이 없어도 시각적으로 다른 유형이 서로 다른 군집으로 갈라지므로,
    언더샘플링 시 특정 유형이 통째로 사라지는 것을 막는 층(stratum) 역할을 함.
    (아토피 / 지루성 / 접촉 피부염처럼 병합된 하위질환 보호가 목적)
    """
    n = len(records)
    if n == 0:
        return 0
    k = max(1, min(k, n // 20))
    if k <= 1:
        for r in records:
            r["vcluster"] = "c00"
        return 1

    # 특징: 64bit pHash(구조) + 색 평균/표준편차 6차원
    feats = np.zeros((n, 70), dtype=np.float32)
    for i, r in enumerate(records):
        v = int(r["phash"], 16)
        feats[i, :64] = [(v >> b) & 1 for b in range(64)]
        feats[i, 64:] = r["color_feat"]
    # 색 특징에 가중치를 줘 구조 비트에 묻히지 않게 함
    feats[:, 64:] *= 8.0

    rng = np.random.default_rng(seed)
    # k-means++ 초기화
    centers = [feats[rng.integers(n)]]
    for _ in range(k - 1):
        d = np.min(
            ((feats[:, None, :] - np.array(centers)[None]) ** 2).sum(-1), axis=1
        )
        s = d.sum()
        idx = rng.integers(n) if s <= 0 else rng.choice(n, p=d / s)
        centers.append(feats[idx])
    C = np.array(centers)

    labels = np.zeros(n, dtype=int)
    for _ in range(25):
        d = ((feats[:, None, :] - C[None]) ** 2).sum(-1)
        new = d.argmin(1)
        if np.array_equal(new, labels):
            break
        labels = new
        for j in range(k):
            m = labels == j
            if m.any():
                C[j] = feats[m].mean(0)

    for r, lab in zip(records, labels):
        r["vcluster"] = f"c{lab:02d}"
    return len(set(labels))


def build_strata(records):
    """층 = 하위질환 × 유사출처 × 시각군집"""
    for r in records:
        r["stratum"] = f"{r['subclass']}||{r['source']}||{r['vcluster']}"


# ══════════════════════════════════════════════════════════
# 2. 근접중복 탐지 (LSH 밴딩) — 전역 수행
# ══════════════════════════════════════════════════════════

def near_duplicate_groups(records, threshold=HASH_THRESHOLD):
    """
    비둘기집 원리: 64bit를 (threshold+1)개 밴드로 나누면
    해밍거리 <= threshold인 두 해시는 최소 한 밴드가 정확히 일치.
    → 같은 밴드값 후보끼리만 비교해 O(n^2) 회피.
    반환: [[idx, ...], ...]  (크기 1인 그룹 포함)
    """
    n = len(records)
    if n == 0:
        return []

    bits = [int(r["phash"], 16) for r in records]
    n_bands = threshold + 1
    band_bits = 64 // n_bands
    mask = (1 << band_bits) - 1

    buckets = defaultdict(list)
    for i, v in enumerate(bits):
        for b in range(n_bands):
            buckets[(b, (v >> (b * band_bits)) & mask)].append(i)

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for idxs in buckets.values():
        # 과대 버킷(단색/균일 이미지 등)은 비교 폭발 방지를 위해 스킵
        if len(idxs) < 2 or len(idxs) > 500:
            continue
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                i, j = idxs[a], idxs[b]
                if find(i) != find(j) and bin(bits[i] ^ bits[j]).count("1") <= threshold:
                    union(i, j)

    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return list(groups.values())


# ══════════════════════════════════════════════════════════
# 3. 층화 언더샘플링
# ══════════════════════════════════════════════════════════

def stratified_sample(records, target, rng, key="stratum"):
    """층(하위질환×유사출처)별 비율을 유지하며 target개 추출."""
    if len(records) <= target:
        return records

    by = defaultdict(list)
    for r in records:
        by[r[key]].append(r)

    total = len(records)
    quota = {s: int(round(target * len(v) / total)) for s, v in by.items()}

    diff = target - sum(quota.values())
    order = sorted(by, key=lambda s: len(by[s]), reverse=True)
    i = 0
    while diff != 0 and order and i < 100000:
        s = order[i % len(order)]
        step = 1 if diff > 0 else -1
        if 0 <= quota[s] + step <= len(by[s]):
            quota[s] += step
            diff -= step
        i += 1

    picked = []
    for s, items in by.items():
        k = min(quota[s], len(items))
        if k <= 0:
            continue
        items = sorted(items, key=lambda r: r["blur"], reverse=True)
        pool = items[: max(k, int(len(items) * 0.7))]  # 선명한 상위 70%에서 랜덤
        picked.extend(rng.sample(pool, k))
    return picked


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="실제 파일 복사 수행")
    args = ap.parse_args()

    rng = random.Random(SEED)
    if not SRC_ROOT.exists():
        sys.exit(f"경로 없음: {SRC_ROOT}")

    # ── 1. 스캔 ──────────────────────────────────────────
    print(f"\n[1/7] 스캔 & 지표 계산   원본: {SRC_ROOT}\n")
    data, broken = {}, []
    for cname in CLASS_DIRS:
        cdir = SRC_ROOT / cname
        if not cdir.exists():
            print(f"  ! 폴더 없음: {cname}")
            data[cname] = []
            continue
        recs = scan_class(cdir, cname)
        data[cname] = [r for r in recs if "error" not in r]
        broken += [r for r in recs if "error" in r]

    # ── 2. 품질 필터 ─────────────────────────────────────
    print(f"\n[2/7] 품질 필터  (MIN_SIDE={MIN_SIDE}, BLUR={BLUR_THRESHOLD})")
    n_lowq = {}
    for cname in CLASS_DIRS:
        recs = data.get(cname, [])
        kept = [r for r in recs
                if r["min_side"] >= MIN_SIDE and r["blur"] >= BLUR_THRESHOLD]
        n_lowq[cname] = len(recs) - len(kept)
        data[cname] = kept
    print("  " + ", ".join(f"{c.split()[0]}:-{n}" for c, n in n_lowq.items()))

    # ── 3. 유사 출처 + 하위질환 ──────────────────────────
    print(f"\n[3/7] 유사 출처 자동 탐지 (해상도 + JPEG 양자화테이블)")
    src_summary = {}
    for cname in CLASS_DIRS:
        counts = assign_pseudo_sources(data.get(cname, []))
        src_summary[cname] = counts
        if not counts:
            continue
        subs = Counter(r["subclass"] for r in data[cname])
        sub_txt = f" | 하위질환 {len(subs)}종" if list(subs) != ["-"] else ""
        top = ", ".join(f"{s}:{n}" for s, n in counts.most_common(3))
        print(f"  {cname:<34s} 출처 {len(counts):2d}개{sub_txt} | {top}")

    # ── 4. 출처 편중 진단 (촬영방식 누출 검사) ───────────
    print(f"\n[4/7] 출처 편중 진단")
    sig_cls = defaultdict(Counter)
    for cname in CLASS_DIRS:
        for r in data.get(cname, []):
            sig_cls[r["raw_sig"]][cname] += 1

    leaks = []
    for sig, cc in sig_cls.items():
        tot = sum(cc.values())
        if tot < LEAK_MIN_IMAGES:
            continue
        cls, n = cc.most_common(1)[0]
        if n / tot >= LEAK_WARN_RATIO:
            leaks.append((sig, cls, n, tot))
    leaks.sort(key=lambda x: -x[2])

    if leaks:
        print(f"  ⚠ 특정 클래스에 몰린 촬영 시그니처 {len(leaks)}건")
        for sig, cls, n, tot in leaks[:8]:
            print(f"     {sig:<34s} → {cls}  {n}/{tot} ({n/tot*100:.0f}%)")
        print("  이 클래스들은 병변이 아니라 '촬영 방식'으로 구분될 수 있음.")
        print("  검증 정확도가 비정상적으로 높으면 이걸 의심할 것.")
    else:
        print("  이상 없음. 촬영 시그니처가 클래스별로 편중되어 있지 않음.")

    # ── 5. 전역 근접중복 ─────────────────────────────────
    print(f"\n[5/7] 전역 근접중복 탐지 (threshold={HASH_THRESHOLD})")
    flat = [r for cname in CLASS_DIRS for r in data.get(cname, [])]
    groups = near_duplicate_groups(flat)

    keep_ids, dup_within, conflicts = set(), 0, []
    for g in groups:
        classes = {flat[i]["cls"] for i in g}
        # 대표: 해상도 큰 것 → 선명한 것
        g_sorted = sorted(
            g, key=lambda i: (flat[i]["min_side"], flat[i]["blur"]), reverse=True
        )
        if len(classes) == 1:
            keep_ids.add(g_sorted[0])
            dup_within += len(g) - 1
        else:
            conflicts.append(g_sorted)
            if CROSS_CLASS_POLICY == "report":
                # 클래스별로 1장씩 남김 (기존 라벨 유지)
                seen = set()
                for i in g_sorted:
                    c = flat[i]["cls"]
                    if c not in seen:
                        seen.add(c)
                        keep_ids.add(i)
            elif CROSS_CLASS_POLICY == "keep_rarest":
                sizes = {c: len(data[c]) for c in classes}
                rarest = min(sizes, key=sizes.get)
                for i in g_sorted:
                    if flat[i]["cls"] == rarest:
                        keep_ids.add(i)
                        break
            # "drop_all" 이면 아무것도 남기지 않음

    n_conf_imgs = sum(len(g) for g in conflicts)
    print(f"  클래스 내 중복 제거: {dup_within}장")
    print(f"  클래스 간 라벨 충돌: {len(conflicts)}그룹 / {n_conf_imgs}장  "
          f"→ 정책={CROSS_CLASS_POLICY}")
    if conflicts and CROSS_CLASS_POLICY == "report":
        print("  ※ 같은 사진이 서로 다른 병명으로 라벨링되어 있음.")
        print("     report.txt 확인 후 drop_all / keep_rarest 로 재실행 권장.")

    surv = defaultdict(list)
    for i in sorted(keep_ids):
        surv[flat[i]["cls"]].append(flat[i])

    # ── 6. 언더샘플링 ────────────────────────────────────
    print(f"\n[6/7] 층화 언더샘플링 (하위질환 × 유사출처 × 시각군집)")
    final, stats = {}, []
    for cname in CLASS_DIRS:
        recs = surv.get(cname, [])
        target = TARGET_COUNTS.get(cname)
        if not target or len(recs) <= target:
            for r in recs:
                r["vcluster"] = "-"
            final[cname] = recs
            stats.append((cname, n_lowq.get(cname, 0), len(recs), len(recs)))
            continue

        print(f"  {cname} — 시각 군집화 중...")
        nk = assign_visual_clusters(recs)
        build_strata(recs)
        sampled = stratified_sample(recs, target, rng)
        final[cname] = sampled
        stats.append((cname, n_lowq.get(cname, 0), len(recs), len(sampled)))

        before = Counter(r["vcluster"] for r in recs)
        after = Counter(r["vcluster"] for r in sampled)
        lost = [c for c in before if after[c] == 0]
        print(f"  {cname:<34s} {len(recs):6d} → {len(sampled):6d}  "
              f"(군집 {nk}개, 소실 군집 {len(lost)}개)")

    total = sum(len(v) for v in final.values())
    print(f"\n  {'클래스':<34s}{'장수':>8s}{'비율':>9s}")
    for cname in CLASS_DIRS:
        n = len(final.get(cname, []))
        print(f"  {cname:<34s}{n:8d}{n/total*100 if total else 0:8.1f}%")
    print(f"  {'합계':<34s}{total:8d}")
    cnts = [len(v) for v in final.values() if v]
    if cnts:
        print(f"  최대/최소 = {max(cnts)/min(cnts):.1f} : 1")
    if broken:
        print(f"  ! 열기 실패 {len(broken)}개")

    if not args.apply:
        print("\n[7/7] dry-run 종료. 실제 복사하려면 --apply 붙여 재실행.\n")
        return

    # ── 7. 복사 ──────────────────────────────────────────
    print(f"\n[7/7] 복사 → {DST_ROOT}")
    DST_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for cname in CLASS_DIRS:
        recs = final.get(cname, [])
        if not recs:
            continue
        outdir = DST_ROOT / cname
        outdir.mkdir(parents=True, exist_ok=True)
        for i, r in enumerate(tqdm(recs, desc=f"  {cname}", ncols=80)):
            src = Path(r["path"])
            dst = outdir / f"{i:06d}{src.suffix.lower()}"
            shutil.copy2(src, dst)
            rows.append({
                "class": cname, "new_path": str(dst), "orig_path": r["path"],
                "subclass": r["subclass"], "pseudo_source": r["source"],
                "width": r["width"], "height": r["height"],
                "vcluster": r.get("vcluster","-"), "blur": round(r["blur"], 2), "phash": r["phash"], "qsig": r["qsig"],
            })

    with open(DST_ROOT / "manifest.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    with open(DST_ROOT / "report.txt", "w", encoding="utf-8") as f:
        f.write("S-CURT 데이터 정제 리포트\n")
        f.write(f"HASH_THRESHOLD={HASH_THRESHOLD} MIN_SIDE={MIN_SIDE} "
                f"BLUR={BLUR_THRESHOLD} POLICY={CROSS_CLASS_POLICY} SEED={SEED}\n\n")
        f.write(f"{'클래스':<34s}{'저품질':>8s}{'중복제거후':>12s}{'최종':>8s}\n")
        for c, lq, sv, fi in stats:
            f.write(f"{c:<34s}{lq:8d}{sv:12d}{fi:8d}\n")
        f.write(f"\n총 {total}장\n")

        f.write(f"\n[출처 편중 경고] {len(leaks)}건\n")
        for sig, cls, n, tot in leaks:
            f.write(f"  {sig:<36s} → {cls}  {n}/{tot} ({n/tot*100:.0f}%)\n")

        f.write(f"\n[유사 출처 분포]\n")
        for cname in CLASS_DIRS:
            f.write(f"{cname}\n")
            for s, n in (src_summary.get(cname) or Counter()).most_common():
                f.write(f"    {s:<40s} {n}\n")

        f.write(f"\n[클래스 간 라벨 충돌] {len(conflicts)}그룹\n")
        for g in conflicts[:300]:
            f.write(f"  그룹 {len(g)}장\n")
            for i in g:
                f.write(f"    [{flat[i]['cls']}] {flat[i]['path']}\n")

        if broken:
            f.write(f"\n[열기 실패] {len(broken)}개\n")
            for b in broken[:300]:
                f.write(f"  {b['path']} — {b['error']}\n")

    print("\n완료. manifest.csv / report.txt 확인.\n")


if __name__ == "__main__":
    main()