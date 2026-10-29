"""
Oculi: Diabetic Retinopathy Screening & Clinical Triage Interface
=================================================================

Interactive research prototype demonstrating:
1. Automated Fundus Image Quality Verification (Sharpness, Illumination, FOV)
2. 5-Grade Diabetic Retinopathy Severity Classification (EfficientNet-B0)
3. Visual Explainability via Grad-CAM Saliency Overlays
4. Evidence-based Clinical Referral & Triage Recommendations
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import streamlit as st

# Ensure repository src is in Python path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from oculi.data.dataset import create_synthetic_fundus_image
from oculi.models.predictor import DR_CLASSES, DRPredictor
from oculi.preprocessing.quality import (
    assess_fundus_quality,
    detect_and_crop_fov,
    to_numpy_rgb,
)

# Page configuration
st.set_page_config(
    page_title="OCULI | Diabetic Retinopathy Screening",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom clinical CSS styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.4);
    }

    .main-title {
        color: #f8fafc;
        font-size: 2.1rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }

    .main-subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-top: 6px;
        margin-bottom: 0;
    }

    .prototype-badge {
        display: inline-block;
        background-color: #3b82f6;
        color: white;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }

    .status-card {
        background-color: #1e293b;
        border-radius: 10px;
        padding: 16px;
        border: 1px solid #334155;
        margin-bottom: 16px;
    }

    .metric-pill-pass {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid #10b981;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }

    .metric-pill-fail {
        background-color: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid #ef4444;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }

    .triage-card {
        border-radius: 10px;
        padding: 18px 20px;
        margin-top: 16px;
    }

    .triage-urgent {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(185, 28, 28, 0.2));
        border: 1px solid #ef4444;
    }

    .triage-referable {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.15), rgba(217, 119, 6, 0.2));
        border: 1px solid #f59e0b;
    }

    .triage-routine {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(5, 150, 105, 0.2));
        border: 1px solid #10b981;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_predictor(checkpoint_path: Optional[str]) -> DRPredictor:
    """Instantiate and cache the deep learning screening predictor by checkpoint path."""
    return DRPredictor(
        checkpoint_path=checkpoint_path,
    )


def main() -> None:
    # Header Banner
    st.markdown(
        """
        <div class="main-header">
            <span class="prototype-badge">Research Prototype</span>
            <h1 class="main-title">OCULI | Diabetic Retinopathy Screening & Triage</h1>
            <p class="main-subtitle">
                Automated Retinal Image Quality Assessment, EfficientNet-B0 Severity Grading,
                Grad-CAM Lesion Explainability, and Clinical Referral Triage.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar: Configurations & Inputs
    with st.sidebar:
        st.header("⚙️ Screening Configuration")

        input_mode = st.radio(
            "Select Fundus Input Source",
            ["Built-in Clinical Demo Samples", "Upload Retinal Fundus Photo"],
            index=0,
        )

        image_to_analyze: Optional[np.ndarray] = None
        sample_name = "Custom Upload"

        if input_mode == "Built-in Clinical Demo Samples":
            demo_grade = st.selectbox(
                "Select Simulated Fundus Case",
                options=[0, 1, 2, 3, 4],
                format_func=lambda g: f"Grade {g}: {DR_CLASSES[g]}",
                index=2,
            )
            sample_name = f"Simulated Grade {demo_grade} ({DR_CLASSES[demo_grade]})"
            image_to_analyze = create_synthetic_fundus_image(size=512, grade=demo_grade, seed=42 + demo_grade)
        else:
            uploaded_file = st.file_uploader(
                "Upload Fundus Photograph",
                type=["png", "jpg", "jpeg"],
                help="Upload high-resolution macula-centered or disc-centered retinal fundus photograph.",
            )
            if uploaded_file is not None:
                pil_img = Image.open(uploaded_file)
                image_to_analyze = np.array(pil_img.convert("RGB"))
                sample_name = uploaded_file.name

        st.markdown("---")
        st.subheader("🔬 Quality Check Thresholds")
        blur_thresh = st.slider("Blur Threshold (Laplacian Var)", 20.0, 200.0, 80.0, 5.0)
        min_illum = st.slider("Min Intensity (Darkness)", 10.0, 60.0, 30.0, 5.0)
        max_illum = st.slider("Max Intensity (Glare)", 180.0, 250.0, 220.0, 5.0)

        st.markdown("---")
        st.subheader("🤖 Model & Calibration")
        custom_checkpoint = st.text_input(
            "Checkpoint Path (optional)",
            value="checkpoints/best_model.pth",
            help="Path to trained .pth weights. If absent, uses ImageNet baseline prototype.",
        )
        temperature = st.slider(
            "Calibration Temperature (T)",
            0.2,
            2.5,
            1.0,
            0.1,
            help="T > 1 softens overconfident logits; T < 1 sharpens confident predictions.",
        )

        chk_path = custom_checkpoint if (custom_checkpoint and Path(custom_checkpoint).exists()) else None

        st.markdown("---")
        st.caption(
            "⚠️ **Disclaimer**: Oculi is a research prototype designed for algorithm "
            "development. It is not FDA/CE cleared for primary clinical diagnosis."
        )

    if image_to_analyze is None:
        st.info("👈 Please select a demo fundus case or upload a fundus photograph from the sidebar to begin.")
        return

    # Load predictor (cached by checkpoint path; temperature scales dynamically without model reload)
    predictor = get_predictor(chk_path)
    predictor.set_temperature(temperature)

    # 1. Image Quality Assessment
    quality_report = assess_fundus_quality(
        image_input=image_to_analyze,
        blur_threshold=blur_thresh,
        min_mean_intensity=min_illum,
        max_mean_intensity=max_illum,
    )

    # Main Tabs
    tab_screening, tab_quality, tab_colab = st.tabs(
        ["🩺 Screening & Triage", "🔍 Quality Diagnostics", "☁️ Google Colab Training Workflow"]
    )

    with tab_screening:
        # Quality Summary Banner
        col_q1, col_q2, col_q3, col_q4 = st.columns(4)
        with col_q1:
            if quality_report.is_acceptable:
                st.markdown("**Diagnostic Quality:** <span class='metric-pill-pass'>PASS (Adequate)</span>", unsafe_allow_html=True)
            else:
                st.markdown("**Diagnostic Quality:** <span class='metric-pill-fail'>NEEDS RECAPTURE</span>", unsafe_allow_html=True)
        with col_q2:
            st.metric("Sharpness Score", f"{quality_report.blur_score:.1f}", f"Threshold: {blur_thresh:.1f}")
        with col_q3:
            st.metric("Mean Illumination", f"{quality_report.mean_intensity:.1f}", quality_report.illumination_status.title())
        with col_q4:
            st.metric("Retinal FOV Ratio", f"{quality_report.fov_ratio * 100:.1f}%", "Aperture Check")

        if quality_report.warnings:
            with st.expander("⚠️ Quality Assessment Warnings", expanded=not quality_report.is_acceptable):
                for w in quality_report.warnings:
                    st.warning(w)
                for r in quality_report.recommendations:
                    st.info(f"💡 Recommendation: {r}")

        st.markdown("---")

        # Run Prediction & Grad-CAM
        try:
            with st.spinner("Analyzing retinal morphology and generating Grad-CAM heatmaps..."):
                pred_result = predictor.predict(image_to_analyze, generate_cam=True)
        except ValueError as err:
            st.error(f"⚠️ Screening Aborted: {err}")
            return

        col_img_orig, col_img_cam = st.columns(2)
        with col_img_orig:
            st.subheader("📷 Fundus Photograph")
            st.image(image_to_analyze, caption=f"Source: {sample_name}", use_container_width=True)

        with col_img_cam:
            st.subheader("🔥 Grad-CAM Lesion Saliency")
            if pred_result.cam_overlay is not None:
                st.image(
                    pred_result.cam_overlay,
                    caption="Grad-CAM Activation Map (Warm colors = High morphological focus)",
                    use_container_width=True,
                )
            else:
                st.info("Grad-CAM generation skipped.")

        st.markdown("---")

        # Classification & Triage Cards
        col_diag, col_triage = st.columns([1, 1])

        with col_diag:
            st.subheader("📊 Severity Classification")
            st.markdown(
                f"### Predicted: **Grade {pred_result.predicted_grade} - {pred_result.predicted_class_name}**"
            )
            st.markdown(f"**Calibrated Model Confidence:** `{pred_result.confidence * 100:.1f}%`")
            st.caption(f"Architecture: {pred_result.model_source}")

            # Probability Breakdown
            st.write("**Class Probability Distribution:**")
            for grade_idx, class_name in DR_CLASSES.items():
                prob = pred_result.probabilities.get(class_name, 0.0)
                st.write(f"Grade {grade_idx} ({class_name}): {prob * 100:.1f}%")
                st.progress(float(prob))

        with col_triage:
            st.subheader("🏥 Clinical Referral Triage")
            triage = pred_result.triage
            card_class = "triage-routine"
            if triage.grade >= 3:
                card_class = "triage-urgent"
            elif triage.grade == 2:
                card_class = "triage-referable"

            rdr_badge = "🔴 REFERABLE DR DETECTED" if triage.is_referable else "🟢 NON-REFERABLE"

            st.markdown(
                f"""
                <div class="triage-card {card_class}">
                    <h4 style="margin: 0; color: #f8fafc;">{rdr_badge}</h4>
                    <p style="margin: 8px 0; font-size: 1.1rem; font-weight: 600; color: #f8fafc;">
                        Recommended Action: {triage.action}
                    </p>
                    <p style="margin: 4px 0; color: #e2e8f0;">
                        <strong>Urgency Timeframe:</strong> {triage.urgency_timeframe}
                    </p>
                    <p style="margin: 4px 0; color: #e2e8f0;">
                        <strong>Clinical Priority:</strong> {triage.priority}
                    </p>
                    <hr style="border-color: rgba(255,255,255,0.1); margin: 12px 0;">
                    <p style="margin: 0; font-size: 0.9rem; color: #cbd5e1;">
                        {triage.clinical_recommendation}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with tab_quality:
        st.subheader("🔬 Fundus Quality Verification Deep Dive")
        st.markdown(
            "Diabetic retinopathy screening algorithms depend critically on high-contrast, "
            "in-focus retinal images. Oculi screens each image across multiple criteria prior to inference."
        )

        col_q_detail1, col_q_detail2 = st.columns(2)
        with col_q_detail1:
            st.write("**Green Channel Edge Energy (Sharpness Analysis)**")
            green = image_to_analyze[:, :, 1]
            lap = cv2.Laplacian(green, cv2.CV_64F)
            lap_norm = np.clip(np.abs(lap), 0, 50) / 50.0
            st.image(lap_norm, caption="Laplacian Edge Filter on Green Channel", use_container_width=True)

        with col_q_detail2:
            st.write("**Automated Field-of-View (FOV) Margin Crop**")
            cropped_fov, bbox = detect_and_crop_fov(image_to_analyze)
            st.image(
                cropped_fov,
                caption=f"Cropped Retinal Disc (BBox: {bbox['ymin']}:{bbox['ymax']}, {bbox['xmin']}:{bbox['xmax']})",
                use_container_width=True,
            )

        st.json(quality_report.to_dict())

    with tab_colab:
        st.subheader("☁️ Google Colab Model Training Guide")
        st.markdown(
            """
            This Oculi codebase is structured for local Windows development while enabling
            seamless GPU training in Google Colab.

            Follow these exact steps to train on Google Colab with free T4 GPU:
            """
        )

        st.markdown(
            """
            #### Step 1: Open Google Colab and Clone / Upload Code
            ```python
            # In Colab notebook cell:
            !git clone https://github.com/<your-username>/OCULI.git
            %cd OCULI
            !pip install -r requirements.txt
            ```

            #### Step 2: Download Dataset (e.g. APTOS 2019 or EyePACS via Kaggle API)
            ```python
            # Set your Kaggle API key
            import os
            os.environ['KAGGLE_CONFIG_DIR'] = '/content'
            !kaggle competitions download -c aptos2019-blindness-detection
            !unzip -q aptos2019-blindness-detection.zip -d data/
            ```

            #### Step 3: Run Model Training with GPU Mixed Precision
            ```bash
            # Train EfficientNet-B0 with automatic checkpointing
            python scripts/train.py \\
                --config configs/default.yaml \\
                --train-csv data/train.csv \\
                --image-dir data/train_images \\
                --epochs 20 \\
                --batch-size 16 \\
                --lr 0.0003
            ```

            #### Step 4: Evaluate Model Checkpoint
            ```bash
            python scripts/evaluate.py \\
                --config configs/default.yaml \\
                --checkpoint checkpoints/best_model.pth \\
                --data-csv data/val.csv \\
                --image-dir data/train_images \\
                --output-json runs/eval_report.json
            ```

            #### Step 5: Export Weights to Local Windows Machine
            Download `checkpoints/best_model.pth` and place it in your local `checkpoints/` folder.
            The Streamlit dashboard will automatically recognize the custom trained weights!
            """
        )


if __name__ == "__main__":
    main()
