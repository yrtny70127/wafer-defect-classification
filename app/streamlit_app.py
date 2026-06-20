"""웨이퍼 맵 결함 분류 + 공정 원인 추론 데모 웹앱.

실행 (프로젝트 루트에서):
    streamlit run app/streamlit_app.py

기능:
    - test set 샘플 선택 또는 웨이퍼 맵 업로드 (.npy / .csv)
    - 결함 패턴 분류 + 공정 원인 추론 + 신뢰도
    - 원인별 조치 가이드 + Grad-CAM 근거 시각화
전제:
    - models/checkpoints/q8_multitask_50ep.pt (09_multitask_longtrain 노트북 실행으로 생성)
    - data/processed/wafer_test.npz (샘플 선택용)
"""
from pathlib import Path

import numpy as np
import streamlit as st
import yaml
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# ---- 경로 (프로젝트 루트 기준 실행 가정) ----
ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "models" / "checkpoints" / "q8_multitask_50ep.pt"
TEST = ROOT / "data" / "processed" / "wafer_test.npz"
MAP_YAML = ROOT / "configs" / "mappings" / "defect_to_cause.yaml"

CLASSES = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc",
           "Near-full", "Random", "Scratch", "none"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# 한글 폰트 (있으면 적용)
def _set_font():
    try:
        import matplotlib
        for name in ["Malgun Gothic", "AppleGothic", "NanumGothic"]:
            if any(name == f.name for f in matplotlib.font_manager.fontManager.ttflist):
                matplotlib.rcParams["font.family"] = name
                break
        matplotlib.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass


class MultiTaskNet(nn.Module):
    def __init__(self, n_defect, n_cause):
        super().__init__()
        bb = models.resnet18(weights=None)
        in_feat = bb.fc.in_features
        bb.fc = nn.Identity()
        self.backbone = bb
        self.defect_head = nn.Linear(in_feat, n_defect)
        self.cause_head = nn.Linear(in_feat, n_cause)

    def forward(self, x):
        f = self.backbone(x)
        return self.defect_head(f), self.cause_head(f)


@st.cache_resource
def load_assets():
    mp = yaml.safe_load(open(MAP_YAML, encoding="utf-8"))
    cause_names = list(mp["causes"]) + ["normal"]
    ck = torch.load(CKPT, map_location=DEVICE)
    net = MultiTaskNet(len(CLASSES), len(cause_names))
    net.load_state_dict(ck["model"])
    net.to(DEVICE).eval()
    return net, mp, cause_names


@st.cache_data
def load_test():
    d = np.load(TEST, allow_pickle=True)
    return d["X"], d["y"]


def to_tensor(wafer_u8):
    img = (wafer_u8.astype(np.float32) / 2.0 - 0.5) / 0.5
    img = np.expand_dims(img, 0).repeat(3, 0)
    return torch.from_numpy(img).unsqueeze(0)


def grad_cam(net, x, target_logit):
    """layer4 기준 Grad-CAM. target_logit: ('defect'|'cause', class_idx)"""
    acts, grads = {}, {}
    h1 = net.backbone.layer4.register_forward_hook(lambda m, i, o: acts.__setitem__("v", o.detach()))
    h2 = net.backbone.layer4.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("v", go[0].detach()))
    net.zero_grad()
    od, oc = net(x)
    head, idx = target_logit
    (od if head == "defect" else oc)[0, idx].backward()
    h1.remove(); h2.remove()
    w = grads["v"].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * acts["v"]).sum(1)).squeeze().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam


def main():
    st.set_page_config(page_title="웨이퍼 결함 진단", layout="wide")
    _set_font()
    st.title("반도체 웨이퍼 결함 패턴 분류 및 공정 원인 추론")
    st.caption("ResNet18 멀티태스크 모델 | 결함 종류 + 공정 원인 + 조치 가이드")

    if not CKPT.exists():
        st.error(f"모델 체크포인트가 없습니다: {CKPT}\n"
                 "먼저 notebooks/09_multitask_longtrain.ipynb 를 실행해 모델을 학습/저장하세요.")
        st.stop()

    net, mp, cause_names = load_assets()
    cause_detail = {k: v.get("detail", "") for k, v in mp["pattern_to_cause"].items()}
    action = mp["action_guide"]

    with st.sidebar:
        st.header("입력 선택")
        mode = st.radio("입력 방식", ["test 샘플 선택", "파일 업로드 (.npy/.csv)"])
        st.markdown("---")
        st.text(f"device: {DEVICE}")
        st.text(f"결함 클래스: {len(CLASSES)}")
        st.text(f"원인 클래스: {len(cause_names)}")

    wafer = None
    true_label = None

    if mode == "test 샘플 선택":
        X, y = load_test()
        only_defect = st.sidebar.checkbox("결함 샘플만 (none 제외)", value=True)
        pool = np.where(y != 8)[0] if only_defect else np.arange(len(X))
        if st.sidebar.button("랜덤 선택"):
            st.session_state["idx"] = int(np.random.choice(pool))
        idx = st.session_state.get("idx", int(pool[0]))
        idx = st.sidebar.number_input("샘플 인덱스", 0, len(X) - 1, idx)
        wafer = X[int(idx)]
        true_label = CLASSES[int(y[int(idx)])]
    else:
        up = st.sidebar.file_uploader("웨이퍼 맵 업로드", type=["npy", "csv"])
        if up is not None:
            if up.name.endswith(".npy"):
                wafer = np.load(up)
            else:
                wafer = np.loadtxt(up, delimiter=",")
            wafer = np.clip(np.rint(wafer), 0, 2).astype(np.uint8)
            if wafer.shape != (128, 128):
                import cv2
                wafer = cv2.resize(wafer, (128, 128), interpolation=cv2.INTER_NEAREST)

    if wafer is None:
        st.info("좌측에서 test 샘플을 선택하거나 웨이퍼 맵 파일을 업로드하세요.")
        return

    # 추론
    x = to_tensor(wafer).to(DEVICE)
    with torch.no_grad():
        od, oc = net(x)
    di = int(od.argmax(1)); ci = int(oc.argmax(1))
    dp = float(od.softmax(1)[0, di]); cp = float(oc.softmax(1)[0, ci])
    defect = CLASSES[di]; cause = cause_names[ci]

    # Grad-CAM
    cam = grad_cam(net, to_tensor(wafer).to(DEVICE), ("defect", di))

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("입력 웨이퍼 맵")
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(wafer, cmap="viridis", vmin=0, vmax=2); ax.axis("off")
        if true_label:
            ax.set_title(f"실제 라벨: {true_label}")
        st.pyplot(fig)

        st.subheader("Grad-CAM (모델이 주목한 영역)")
        fig2, ax2 = plt.subplots(figsize=(4, 4))
        ax2.imshow(wafer / 2.0, cmap="gray")
        ax2.imshow(cam, cmap="jet", alpha=0.5, extent=(0, 128, 128, 0), interpolation="bilinear")
        ax2.axis("off")
        st.pyplot(fig2)

    with col2:
        st.subheader("진단 리포트")
        st.metric("결함 패턴", defect, f"{dp*100:.1f}% 확신")
        st.metric("공정 원인", cause, f"{cp*100:.1f}% 확신")
        if defect in cause_detail:
            st.markdown(f"**상세 원인:** {cause_detail[defect]}")
        if cause in action:
            st.success(f"**조치 가이드:** {action[cause]}")
        if true_label:
            ok = (true_label == defect)
            st.markdown(("정답과 일치 ✅" if ok else "정답과 불일치 ⚠️") + f" (실제: {true_label})")

        st.markdown("---")
        st.caption("결함 종류별 예측 확률")
        probs = od.softmax(1)[0].detach().cpu().numpy()
        order = probs.argsort()[::-1][:5]
        for j in order:
            st.write(f"{CLASSES[j]}: {probs[j]*100:.1f}%")


if __name__ == "__main__":
    main()
