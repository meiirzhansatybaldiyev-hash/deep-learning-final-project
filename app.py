# ============================================================
# streamlit_app.py — Plant Disease Model Comparison Dashboard
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io

# --------------------------------------------------
# Page config
# --------------------------------------------------

st.set_page_config(
    page_title="Plant Disease — Model Comparison",
    page_icon="🌿",
    layout="wide"
)

# --------------------------------------------------
# Load saved results
# --------------------------------------------------

@st.cache_data
def load_data():
    comparison = pd.read_csv("model_comparison.csv")
    losses     = pd.read_csv("training_losses.csv")
    per_class  = pd.read_csv("per_class_metrics.csv")
    cm_w2      = np.load("cm_baseline_cnn.npy")
    cm_w3      = np.load("cm_resnet_feature_extract.npy")
    cm_w4      = np.load("cm_resnet_finetune.npy")

    with open("class_names.json") as f:
        class_names = json.load(f)

    return comparison, losses, per_class, cm_w2, cm_w3, cm_w4, class_names

comparison, losses, per_class, cm_w2, cm_w3, cm_w4, class_names = load_data()

MODEL_CMS = {
    "Week 2 — Baseline CNN":                cm_w2,
    "Week 3 — ResNet18 Feature Extraction": cm_w3,
    "Week 4 — ResNet18 Full Fine-tuning":   cm_w4,
}

# --------------------------------------------------
# Model definitions & loaders
# --------------------------------------------------

class BaselineCNN(nn.Module):
    def __init__(self, num_classes):
        super(BaselineCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


@st.cache_resource
def load_model_w2(num_classes):
    model = BaselineCNN(num_classes=num_classes)
    model.load_state_dict(torch.load("baseline_cnn.pth", map_location="cpu"))
    model.eval()
    return model


@st.cache_resource
def load_model_w3(num_classes):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load("resnet_feature_extract.pth", map_location="cpu"))
    model.eval()
    return model


@st.cache_resource
def load_model_w4(num_classes):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load("final_resnet_plant_disease.pth", map_location="cpu"))
    model.eval()
    return model


# --------------------------------------------------
# Transforms
# --------------------------------------------------

transform_128 = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

transform_224 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# --------------------------------------------------
# Predict function
# --------------------------------------------------

def predict(model, image: Image.Image, transform, class_names, top_k=5):
    tensor = transform(image).unsqueeze(0)  # (1, C, H, W)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]

    top_probs, top_idxs = torch.topk(probs, k=top_k)

    results = [
        {
            "class":       class_names[idx.item()],
            "probability": round(prob.item() * 100, 2)
        }
        for prob, idx in zip(top_probs, top_idxs)
    ]

    return results


# --------------------------------------------------
# Format class name for display
# --------------------------------------------------

def format_label(raw: str) -> str:
    # e.g. "Tomato___Early_blight" -> "Tomato — Early Blight"
    raw = raw.replace("___", " — ").replace("_", " ")
    return raw.title()


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.title("🌿 Plant Disease Classifier")
st.sidebar.markdown("Compare three models trained over 4 weeks.")

page = st.sidebar.radio(
    "Navigate",
    ["🔬 Predict", "📊 Overview", "📉 Training Curves",
     "🔍 Per-Class Metrics", "🔲 Confusion Matrix"]
)


# ============================================================
# PAGE 0 — Predict
# ============================================================

if page == "🔬 Predict":

    st.title("🔬 Predict Plant Disease from Image")
    st.markdown("Upload a leaf photo and get predictions from all three models.")

    uploaded = st.file_uploader(
        "Upload a leaf image",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded:
        image = Image.open(uploaded).convert("RGB")

        col_img, col_info = st.columns([1, 2])

        with col_img:
            st.image(image, caption="Uploaded image", use_container_width=True)

        with col_info:
            st.markdown("### Image Info")
            st.write(f"**Size:** {image.width} × {image.height} px")
            st.write(f"**File:** {uploaded.name}")
            st.write(f"**Format:** {uploaded.type}")

        st.markdown("---")

        # --- Load models ---

        num_classes = len(class_names)

        try:
            model_w2 = load_model_w2(num_classes)
            w2_loaded = True
        except FileNotFoundError:
            w2_loaded = False

        try:
            model_w3 = load_model_w3(num_classes)
            w3_loaded = True
        except FileNotFoundError:
            w3_loaded = False

        try:
            model_w4 = load_model_w4(num_classes)
            w4_loaded = True
        except FileNotFoundError:
            w4_loaded = False

        # --- Run predictions ---

        model_configs = [
            ("Week 2 — Baseline CNN",                w2_loaded, lambda: load_model_w2(num_classes), transform_128),
            ("Week 3 — ResNet18 Feature Extraction", w3_loaded, lambda: load_model_w3(num_classes), transform_224),
            ("Week 4 — ResNet18 Full Fine-tuning",   w4_loaded, lambda: load_model_w4(num_classes), transform_224),
        ]

        cols = st.columns(3)

        for col, (name, loaded, get_model, transform) in zip(cols, model_configs):
            with col:
                st.markdown(f"#### {name}")

                if not loaded:
                    st.warning("Model file not found.\nRun the training notebook first.")
                    continue

                model = get_model()
                preds = predict(model, image, transform, class_names, top_k=5)

                # Top prediction highlighted
                top = preds[0]
                confidence = top["probability"]
                color = "#57B894" if confidence >= 70 else "#F4845F" if confidence < 40 else "#5C85D6"

                st.markdown(
                    f"""
                    <div style="border:2px solid {color}; border-radius:10px;
                                padding:14px; text-align:center; margin-bottom:12px">
                        <div style="font-size:0.8rem; color:#888">Top prediction</div>
                        <div style="font-size:1.1rem; font-weight:bold; margin:6px 0">
                            {format_label(top['class'])}
                        </div>
                        <div style="font-size:2rem; font-weight:bold; color:{color}">
                            {confidence}%
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Bar chart for top-5
                fig, ax = plt.subplots(figsize=(4, 3))

                labels_chart = [format_label(p["class"]) for p in reversed(preds)]
                probs_chart  = [p["probability"] for p in reversed(preds)]
                bar_colors   = [color if i == len(preds) - 1 else "#CCCCCC"
                                for i in range(len(preds))]

                ax.barh(labels_chart, probs_chart, color=bar_colors)
                ax.set_xlim(0, 105)
                ax.set_xlabel("Confidence (%)")
                ax.set_title("Top 5 predictions")

                for i, v in enumerate(probs_chart):
                    ax.text(v + 1, i, f"{v}%", va="center", fontsize=8)

                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

        # --- Side-by-side comparison of top predictions ---

        st.markdown("---")
        st.subheader("📋 Summary — Top Predictions")

        summary_rows = []

        for name, loaded, get_model, transform in model_configs:
            if loaded:
                model = get_model()
                preds = predict(model, image, transform, class_names, top_k=1)
                summary_rows.append({
                    "Model":       name,
                    "Prediction":  format_label(preds[0]["class"]),
                    "Confidence":  f"{preds[0]['probability']}%",
                })
            else:
                summary_rows.append({
                    "Model":      name,
                    "Prediction": "—",
                    "Confidence": "—",
                })

        st.dataframe(
            pd.DataFrame(summary_rows),
            use_container_width=True,
            hide_index=True
        )

    else:
        # Placeholder when no image is uploaded
        st.info("👆 Upload a leaf image above to get started.")

        st.markdown("### Supported classes (examples)")

        sample_classes = [format_label(c) for c in class_names[:12]]
        cols = st.columns(3)

        for i, cls in enumerate(sample_classes):
            cols[i % 3].markdown(f"- {cls}")

        if len(class_names) > 12:
            st.markdown(f"*... and {len(class_names) - 12} more classes*")


# ============================================================
# PAGE 1 — Overview
# ============================================================

elif page == "📊 Overview":

    st.title("📊 Model Comparison — Overview")
    st.markdown("Overall performance of each model on the **test set**.")

    best_acc_idx = comparison["accuracy"].idxmax()

    cols = st.columns(len(comparison))

    for i, row in comparison.iterrows():
        with cols[i]:
            is_best = (i == best_acc_idx)
            border  = "2px solid #4CAF50" if is_best else "1px solid #ddd"
            badge   = " 🏆" if is_best else ""

            st.markdown(
                f"""
                <div style="border:{border}; border-radius:10px; padding:16px; text-align:center">
                    <b>{row['model']}{badge}</b><br><br>
                    <span style="font-size:2rem; color:#4CAF50"><b>{row['accuracy']}%</b></span><br>
                    <small>Accuracy</small><br><br>
                    F1&nbsp;&nbsp;&nbsp;<b>{row['f1_weighted']}%</b><br>
                    Precision&nbsp;&nbsp;&nbsp;<b>{row['precision_weighted']}%</b><br>
                    Recall&nbsp;&nbsp;&nbsp;<b>{row['recall_weighted']}%</b>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("---")

    st.subheader("Metrics Comparison")

    metrics = ["accuracy", "f1_weighted", "precision_weighted", "recall_weighted"]
    labels  = ["Accuracy", "F1 (weighted)", "Precision (weighted)", "Recall (weighted)"]

    x      = np.arange(len(metrics))
    width  = 0.25
    colors = ["#5C85D6", "#F4845F", "#57B894"]

    fig, ax = plt.subplots(figsize=(11, 5))

    for i, (_, row) in enumerate(comparison.iterrows()):
        vals = [row[m] for m in metrics]
        bars = ax.bar(x + i * width, vals, width, label=row["model"], color=colors[i])
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{bar.get_height():.1f}%",
                ha="center", va="bottom", fontsize=8
            )

    ax.set_xticks(x + width)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score (%)")
    ax.set_title("Overall Metrics — All Models")
    ax.legend()
    ax.set_ylim(0, 115)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    st.pyplot(fig)

    st.markdown("---")
    st.subheader("Raw Numbers")

    display_df = comparison.rename(columns={
        "model":              "Model",
        "accuracy":           "Accuracy (%)",
        "f1_weighted":        "F1 Weighted (%)",
        "precision_weighted": "Precision (%)",
        "recall_weighted":    "Recall (%)",
    })

    st.dataframe(display_df, use_container_width=True, hide_index=True)


# ============================================================
# PAGE 2 — Training Curves
# ============================================================

elif page == "📉 Training Curves":

    st.title("📉 Training Loss Curves")
    st.markdown("Loss per epoch for each model.")

    model_cols = [c for c in losses.columns if c != "epoch"]

    selected = st.multiselect(
        "Select models to display",
        options=model_cols,
        default=model_cols
    )

    if selected:
        colors    = ["#5C85D6", "#F4845F", "#57B894"]
        color_map = {m: colors[i % len(colors)] for i, m in enumerate(model_cols)}

        fig, ax = plt.subplots(figsize=(10, 5))

        for col in selected:
            vals = losses[col].dropna()
            ax.plot(range(1, len(vals) + 1), vals,
                    marker="o", label=col, color=color_map[col], linewidth=2)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Training Loss")
        ax.set_title("Training Loss per Epoch")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()

        st.pyplot(fig)

        st.markdown("---")
        st.subheader("Raw Loss Values")
        st.dataframe(losses[["epoch"] + selected], use_container_width=True, hide_index=True)

    else:
        st.info("Select at least one model above.")


# ============================================================
# PAGE 3 — Per-Class Metrics
# ============================================================

elif page == "🔍 Per-Class Metrics":

    st.title("🔍 Per-Class Metrics")
    st.markdown("Precision, Recall, and F1 broken down by plant disease class.")

    model_choice  = st.selectbox("Select model", options=per_class["model"].unique())
    metric_choice = st.radio("Metric", ["f1", "precision", "recall"], horizontal=True)

    filtered = per_class[per_class["model"] == model_choice].sort_values(
        metric_choice, ascending=True
    )

    fig, ax = plt.subplots(figsize=(10, max(6, len(filtered) * 0.35)))

    colors_bar = [
        "#57B894" if v >= 80 else "#F4845F" if v < 50 else "#5C85D6"
        for v in filtered[metric_choice]
    ]

    ax.barh(filtered["class"], filtered[metric_choice], color=colors_bar)
    ax.set_xlabel(f"{metric_choice.capitalize()} (%)")
    ax.set_title(f"{metric_choice.capitalize()} per Class — {model_choice}")
    ax.axvline(80, color="green", linestyle="--", alpha=0.5, label="80% threshold")
    ax.axvline(50, color="red",   linestyle="--", alpha=0.5, label="50% threshold")
    ax.legend()
    ax.set_xlim(0, 110)
    plt.tight_layout()

    st.pyplot(fig)

    st.markdown("---")
    st.subheader("Full Table")

    search = st.text_input("Filter by class name")
    table  = filtered[["class", "precision", "recall", "f1", "support"]]

    if search:
        table = table[table["class"].str.contains(search, case=False)]

    st.dataframe(table, use_container_width=True, hide_index=True)


# ============================================================
# PAGE 4 — Confusion Matrix
# ============================================================

elif page == "🔲 Confusion Matrix":

    st.title("🔲 Confusion Matrix")

    model_choice = st.selectbox("Select model", options=list(MODEL_CMS.keys()))

    cm        = MODEL_CMS[model_choice]
    normalize = st.toggle("Normalize (row %)", value=False)

    if normalize:
        cm_plot = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
        fmt     = ".2f"
    else:
        cm_plot = cm
        fmt     = "d"

    fig_size  = st.slider("Figure size", min_value=6, max_value=20, value=12)

    fig, ax   = plt.subplots(figsize=(fig_size, fig_size))
    show_annot = len(class_names) <= 20

    sns.heatmap(
        cm_plot,
        cmap="Blues",
        ax=ax,
        annot=show_annot,
        fmt=fmt if show_annot else "",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.3 if show_annot else 0
    )

    ax.set_title(f"Confusion Matrix — {model_choice}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    st.pyplot(fig)

    st.markdown("---")

    correct = np.trace(cm)
    total   = cm.sum()

    c1, c2, c3 = st.columns(3)
    c1.metric("Total samples", f"{total:,}")
    c2.metric("Correct",       f"{correct:,}")
    c3.metric("Accuracy",      f"{100 * correct / total:.2f}%")