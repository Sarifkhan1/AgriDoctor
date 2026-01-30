"""
AgriDoctor AI - Streamlit Annotation Tool
Image viewer with labeling controls for crop disease classification.
"""

import streamlit as st
import pandas as pd
import os
import json
import uuid
from datetime import datetime
from pathlib import Path
from PIL import Image
import io

# Page config
st.set_page_config(
    page_title="AgriDoctor Annotator",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UX
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a5a2e 0%, #2d8a4e 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
        text-align: center;
    }
    .main-header h1 {
        margin: 0;
        font-size: 2rem;
    }
    .stat-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        border-left: 4px solid #2d8a4e;
    }
    .stat-number {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1a5a2e;
    }
    .stat-label {
        font-size: 0.9rem;
        color: #666;
    }
    .image-container {
        border: 2px solid #ddd;
        border-radius: 12px;
        overflow: hidden;
        background: #fafafa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
    }
    .keyboard-hint {
        background: #f0f0f0;
        padding: 0.3rem 0.6rem;
        border-radius: 4px;
        font-family: monospace;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Constants
CROPS = ["tomato", "potato", "rice", "maize", "chili", "cucumber"]

DISEASE_LABELS = {
    "tomato": [
        "TOM_EARLY_BLIGHT", "TOM_LATE_BLIGHT", "TOM_LEAF_MOLD", "TOM_SEPTORIA",
        "TOM_SPIDER_MITES", "TOM_MOSAIC", "TOM_BACL_SPOT", "TOM_BLOSSOM_ROT",
        "TOM_HEALTHY", "TOM_UNKNOWN"
    ],
    "potato": [
        "POT_EARLY_BLIGHT", "POT_LATE_BLIGHT", "POT_BLACKLEG", "POT_SCAB",
        "POT_VIRAL", "POT_APHIDS", "POT_HEALTHY", "POT_UNKNOWN"
    ],
    "rice": [
        "RICE_BLAST", "RICE_BROWN_SPOT", "RICE_BACT_BLIGHT", "RICE_TUNGRO",
        "RICE_SHEATH_BLIGHT", "RICE_STEMB", "RICE_HEALTHY", "RICE_UNKNOWN"
    ],
    "maize": [
        "MAIZE_NLB", "MAIZE_RUST", "MAIZE_GLS", "MAIZE_SMUT",
        "MAIZE_BORER", "MAIZE_HEALTHY", "MAIZE_UNKNOWN"
    ],
    "chili": [
        "CHILI_ANTHRAC", "CHILI_BACT_WILT", "CHILI_LEAF_CURL", "CHILI_POWDERY",
        "CHILI_THRIPS", "CHILI_HEALTHY", "CHILI_UNKNOWN"
    ],
    "cucumber": [
        "CUC_POWDERY", "CUC_DOWNY", "CUC_ANGULAR", "CUC_MOSAIC",
        "CUC_ANTHRAC", "CUC_APHIDS", "CUC_HEALTHY", "CUC_UNKNOWN"
    ]
}

CATEGORY_CODES = {
    "F": "Fungal", "B": "Bacterial", "V": "Viral",
    "P": "Pest", "N": "Nutrient", "H": "Healthy", "U": "Unknown"
}


def get_base_path():
    """Get the base path for data storage."""
    return Path(__file__).parent.parent / "data"


def init_labels_csv(labels_path: Path):
    """Initialize labels.csv if it doesn't exist."""
    if not labels_path.exists():
        labels_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(columns=[
            "label_id", "image_path", "encounter_id", "crop",
            "primary_label", "secondary_labels", "severity_score",
            "quality_score", "labeler_id", "labeled_at", "qa_verified", "notes"
        ])
        df.to_csv(labels_path, index=False)


def load_labels(labels_path: Path) -> pd.DataFrame:
    """Load existing labels from CSV."""
    if labels_path.exists():
        return pd.read_csv(labels_path)
    return pd.DataFrame()


def save_label(labels_path: Path, label_data: dict):
    """Append a new label to the CSV file."""
    df = load_labels(labels_path)
    new_df = pd.DataFrame([label_data])
    df = pd.concat([df, new_df], ignore_index=True)
    df.to_csv(labels_path, index=False)


def get_image_files(image_dir: Path, crop: str = None) -> list:
    """Get list of image files to label."""
    images = []
    extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    
    if crop:
        search_path = image_dir / "crops" / crop
    else:
        search_path = image_dir / "crops"
    
    if search_path.exists():
        for ext in extensions:
            images.extend(search_path.rglob(f"*{ext}"))
    
    return sorted(images)


def get_unlabeled_images(all_images: list, labeled_paths: set) -> list:
    """Filter out already labeled images."""
    return [img for img in all_images if str(img) not in labeled_paths]


def main():
    # Header
    st.markdown("""
        <div class="main-header">
            <h1>🌿 AgriDoctor Annotation Tool</h1>
            <p>Label crop disease images for AI training</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Initialize paths
    base_path = get_base_path()
    labels_path = base_path / "labels" / "labels.csv"
    image_dir = base_path / "images"
    
    # Initialize
    init_labels_csv(labels_path)
    
    # Sidebar - Settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Labeler ID
        labeler_id = st.text_input(
            "Labeler ID",
            value=st.session_state.get("labeler_id", ""),
            help="Your unique identifier"
        )
        st.session_state.labeler_id = labeler_id
        
        # Crop filter
        crop_filter = st.selectbox(
            "Filter by Crop",
            options=["All"] + CROPS,
            index=0
        )
        
        st.divider()
        
        # Statistics
        st.header("📊 Statistics")
        labels_df = load_labels(labels_path)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number">{len(labels_df)}</div>
                    <div class="stat-label">Labeled</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            today_count = len(labels_df[labels_df.get('labeled_at', '').str.contains(datetime.now().strftime('%Y-%m-%d'), na=False)]) if not labels_df.empty else 0
            st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number">{today_count}</div>
                    <div class="stat-label">Today</div>
                </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # Keyboard shortcuts
        st.header("⌨️ Shortcuts")
        st.markdown("""
        - <span class="keyboard-hint">S</span> Save & Next
        - <span class="keyboard-hint">→</span> Next image
        - <span class="keyboard-hint">←</span> Previous image
        - <span class="keyboard-hint">H</span> Mark Healthy
        """, unsafe_allow_html=True)
    
    # Get images
    selected_crop = None if crop_filter == "All" else crop_filter
    all_images = get_image_files(image_dir, selected_crop)
    
    if not all_images:
        st.warning(f"No images found in {image_dir}/crops/")
        st.info("Upload images to the appropriate crop folder to start labeling.")
        
        # Upload section
        st.header("📤 Upload Images")
        uploaded_crop = st.selectbox("Select crop for upload", CROPS)
        uploaded_files = st.file_uploader(
            "Upload crop images",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True
        )
        
        if uploaded_files and st.button("Save Uploaded Images"):
            upload_path = image_dir / "crops" / uploaded_crop
            upload_path.mkdir(parents=True, exist_ok=True)
            
            for file in uploaded_files:
                file_path = upload_path / file.name
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
            
            st.success(f"Saved {len(uploaded_files)} images to {upload_path}")
            st.rerun()
        
        return
    
    # Get labeled image paths
    labels_df = load_labels(labels_path)
    labeled_paths = set(labels_df['image_path'].tolist()) if not labels_df.empty else set()
    
    # Filter to unlabeled
    unlabeled_images = get_unlabeled_images(all_images, labeled_paths)
    
    # Image navigation
    if "image_index" not in st.session_state:
        st.session_state.image_index = 0
    
    # Progress bar
    progress = len(labeled_paths) / len(all_images) if all_images else 0
    st.progress(progress, text=f"Progress: {len(labeled_paths)}/{len(all_images)} images labeled ({progress:.1%})")
    
    if not unlabeled_images:
        st.success("🎉 All images have been labeled!")
        return
    
    # Current image
    current_idx = min(st.session_state.image_index, len(unlabeled_images) - 1)
    current_image_path = unlabeled_images[current_idx]
    
    # Layout
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.subheader(f"📷 Image {current_idx + 1} of {len(unlabeled_images)}")
        
        # Image display with zoom
        try:
            image = Image.open(current_image_path)
            st.image(image, use_container_width=True)
            st.caption(f"📁 {current_image_path.name}")
        except Exception as e:
            st.error(f"Error loading image: {e}")
        
        # Navigation buttons
        nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
        with nav_col1:
            if st.button("⬅️ Previous", disabled=current_idx == 0):
                st.session_state.image_index = max(0, current_idx - 1)
                st.rerun()
        with nav_col3:
            if st.button("Next ➡️", disabled=current_idx >= len(unlabeled_images) - 1):
                st.session_state.image_index = min(len(unlabeled_images) - 1, current_idx + 1)
                st.rerun()
    
    with col_right:
        st.subheader("🏷️ Label This Image")
        
        # Detect crop from path
        detected_crop = None
        for crop in CROPS:
            if crop in str(current_image_path).lower():
                detected_crop = crop
                break
        
        # Crop selection
        crop = st.selectbox(
            "Crop Type",
            options=CROPS,
            index=CROPS.index(detected_crop) if detected_crop else 0
        )
        
        # Primary label
        available_labels = DISEASE_LABELS.get(crop, [])
        primary_label = st.selectbox(
            "Primary Label *",
            options=available_labels,
            index=0
        )
        
        # Secondary labels (multi-select)
        secondary_options = [l for l in available_labels if l != primary_label]
        secondary_labels = st.multiselect(
            "Secondary Labels (optional)",
            options=secondary_options,
            max_selections=3
        )
        
        st.divider()
        
        # Severity score
        severity_score = st.slider(
            "Severity Score",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="0 = Healthy, 1 = Critical"
        )
        
        # Severity guide
        if severity_score < 0.2:
            st.info("💚 Very mild - minimal symptoms")
        elif severity_score < 0.4:
            st.info("💛 Mild - scattered symptoms")
        elif severity_score < 0.6:
            st.warning("🟠 Moderate - multiple leaves affected")
        elif severity_score < 0.8:
            st.warning("🔴 Severe - spreading damage")
        else:
            st.error("⚫ Critical - plant viability threatened")
        
        # Quality score
        quality_score = st.slider(
            "Image Quality Score",
            min_value=0.0,
            max_value=1.0,
            value=0.8,
            step=0.05,
            help="Image clarity, focus, lighting"
        )
        
        # Notes
        notes = st.text_area(
            "Notes (optional)",
            placeholder="Any observations, edge cases, or comments..."
        )
        
        st.divider()
        
        # Save button
        if st.button("💾 Save Label", type="primary", use_container_width=True):
            if not labeler_id:
                st.error("Please enter your Labeler ID in the sidebar!")
            else:
                label_data = {
                    "label_id": str(uuid.uuid4()),
                    "image_path": str(current_image_path),
                    "encounter_id": current_image_path.stem.split("_")[1] if "_" in current_image_path.stem else "",
                    "crop": crop,
                    "primary_label": primary_label,
                    "secondary_labels": json.dumps(secondary_labels),
                    "severity_score": severity_score,
                    "quality_score": quality_score,
                    "labeler_id": labeler_id,
                    "labeled_at": datetime.now().isoformat(),
                    "qa_verified": False,
                    "notes": notes
                }
                
                save_label(labels_path, label_data)
                st.success("✅ Label saved!")
                
                # Move to next image
                if current_idx < len(unlabeled_images) - 1:
                    st.session_state.image_index = current_idx + 1
                st.rerun()
        
        # Quick actions
        st.divider()
        col_q1, col_q2 = st.columns(2)
        with col_q1:
            if st.button("🌿 Quick: Healthy", use_container_width=True):
                healthy_label = f"{crop.upper()[:3]}_HEALTHY" if len(crop) >= 3 else f"{crop.upper()}_HEALTHY"
                # Find matching healthy label
                for label in available_labels:
                    if "HEALTHY" in label:
                        healthy_label = label
                        break
                
                label_data = {
                    "label_id": str(uuid.uuid4()),
                    "image_path": str(current_image_path),
                    "encounter_id": current_image_path.stem.split("_")[1] if "_" in current_image_path.stem else "",
                    "crop": crop,
                    "primary_label": healthy_label,
                    "secondary_labels": "[]",
                    "severity_score": 0.0,
                    "quality_score": quality_score,
                    "labeler_id": labeler_id if labeler_id else "quick",
                    "labeled_at": datetime.now().isoformat(),
                    "qa_verified": False,
                    "notes": "Quick healthy label"
                }
                save_label(labels_path, label_data)
                st.success("✅ Marked as Healthy!")
                if current_idx < len(unlabeled_images) - 1:
                    st.session_state.image_index = current_idx + 1
                st.rerun()
        
        with col_q2:
            if st.button("❓ Quick: Unknown", use_container_width=True):
                unknown_label = f"{crop.upper()[:3]}_UNKNOWN" if len(crop) >= 3 else f"{crop.upper()}_UNKNOWN"
                for label in available_labels:
                    if "UNKNOWN" in label:
                        unknown_label = label
                        break
                
                label_data = {
                    "label_id": str(uuid.uuid4()),
                    "image_path": str(current_image_path),
                    "encounter_id": current_image_path.stem.split("_")[1] if "_" in current_image_path.stem else "",
                    "crop": crop,
                    "primary_label": unknown_label,
                    "secondary_labels": "[]",
                    "severity_score": 0.5,
                    "quality_score": quality_score,
                    "labeler_id": labeler_id if labeler_id else "quick",
                    "labeled_at": datetime.now().isoformat(),
                    "qa_verified": False,
                    "notes": "Quick unknown label - needs review"
                }
                save_label(labels_path, label_data)
                st.warning("⚠️ Marked as Unknown")
                if current_idx < len(unlabeled_images) - 1:
                    st.session_state.image_index = current_idx + 1
                st.rerun()


if __name__ == "__main__":
    main()
