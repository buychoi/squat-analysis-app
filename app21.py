import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import csv
import datetime
import time
import os
import platform
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import requests
import json
from io import BytesIO
import traceback
import sys
from openai import OpenAI

# Print debugging information
print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")
print(f"Platform: {sys.platform}")

# -----------------------------------------------------------------------------------
# 1) Path Settings - Cross-platform compatible
# -----------------------------------------------------------------------------------
# Detect operating system
SYSTEM = platform.system()  # 'Windows', 'Darwin'(Mac), 'Linux'
print(f"Operating system: {SYSTEM}")

# Default base directory - set to your actual project path
if SYSTEM == 'Windows':
    DEFAULT_BASE_DIR = r"/Users/taeseokchoi/ALL/TWH/POS/pose_streamlit(04)"
else:
    # For macOS or Linux, use the specified path
    DEFAULT_BASE_DIR = r"/Users/taeseokchoi/ALL/TWH/POS/pose_streamlit(04)"

# Set base directory in Streamlit session state
if 'base_dir' not in st.session_state:
    st.session_state.base_dir = DEFAULT_BASE_DIR

# Debug output
print(f"Base directory set to: {st.session_state.base_dir}")

# Define required folder paths
USERS_DIR = os.path.join(st.session_state.base_dir, "users")
STANDARD_IMG_DIR = os.path.join(st.session_state.base_dir, "imagestandard")
LOGO_PATH = os.path.join(st.session_state.base_dir, "logo", "healthnai_logo.png")

print(f"USERS_DIR: {USERS_DIR}")
print(f"STANDARD_IMG_DIR: {STANDARD_IMG_DIR}")
print(f"LOGO_PATH: {LOGO_PATH}")

# Create directories
os.makedirs(st.session_state.base_dir, exist_ok=True)
os.makedirs(USERS_DIR, exist_ok=True)
os.makedirs(STANDARD_IMG_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOGO_PATH), exist_ok=True)

# Verify directories
required_paths = [
    st.session_state.base_dir,
    USERS_DIR,
    STANDARD_IMG_DIR,
    os.path.dirname(LOGO_PATH)
]
for path in required_paths:
    if not os.path.exists(path):
        print(f"Creating path: {path}")
        os.makedirs(path, exist_ok=True)
    else:
        print(f"Path verified: {path}")

# -----------------------------------------------------------------------------------
# 2) OpenAI API Setup
# -----------------------------------------------------------------------------------
# Set your OpenAI API key here (or use environment variable)
# Note: You should replace this with your valid OpenAI API key
OPENAI_API_KEY = "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"  # Enter your valid API key here
print(f"OpenAI API key set: {'Yes' if OPENAI_API_KEY else 'No'}")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------------------------------------------------------------
# 3) Mediapipe Setup
# -----------------------------------------------------------------------------------
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_drawing_styles = mp.solutions.drawing_styles

# -----------------------------------------------------------------------------------
# 4) Page Configuration
# -----------------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Squat Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stButton > button {
        width: 100%;
        height: 3em;
        font-size: 1.1em;
        font-weight: bold;
    }
    .main > div {
        padding-left: 2rem;
        padding-right: 2rem;
    }
    h1, h2, h3 {
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .stExpander {
        border: 1px solid #e6e6e6;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .logo-container {
        display: flex;
        justify-content: center;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------------
# 5) Video Codec Setup - Platform-specific support
# -----------------------------------------------------------------------------------
def get_video_codec():
    """Return appropriate video codec for the platform"""
    if SYSTEM == 'Windows':
        # Windows codec options to try
        codecs_to_try = ['XVID', 'MJPG', 'H264', 'X264', 'WMV1']
        for codec in codecs_to_try:
            try:
                codec_value = cv2.VideoWriter_fourcc(*codec)
                print(f"Windows: {codec} codec success ({codec_value})")
                return codec_value
            except Exception as e:
                print(f"Windows: {codec} codec failed: {str(e)}")
                continue
        # Fallback
        print("Windows: Using fallback MJPG codec")
        return cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    else:  # Mac/Linux
        try:
            codec_value = cv2.VideoWriter_fourcc(*'mp4v')  # Recommended for macOS
            print(f"Mac/Linux: mp4v codec success ({codec_value})")
            return codec_value
        except Exception as e:
            print(f"Mac/Linux: mp4v codec failed: {str(e)}")
            try:
                codec_value = cv2.VideoWriter_fourcc(*'XVID')
                print(f"Mac/Linux: XVID codec success ({codec_value})")
                return codec_value
            except Exception as e:
                print(f"Mac/Linux: XVID codec failed: {str(e)}")
                print("Mac/Linux: Using fallback MJPG codec")
                return cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')

# -----------------------------------------------------------------------------------
# 6) Session State Reset Function
# -----------------------------------------------------------------------------------
def reset_session_state(keep_user=True):
    """Reset session state variables (optionally keeping user data)"""
    print(f"Starting session state reset (keep user: {keep_user})")
    current_user = None
    users = {}
    if keep_user and 'current_user' in st.session_state:
        current_user = st.session_state.current_user
    if keep_user and 'users' in st.session_state:
        users = st.session_state.users
    
    st.session_state.capture_running = False
    st.session_state.csv_file = None
    st.session_state.csv_writer = None
    st.session_state.current_squat_count = 0
    st.session_state.squat_positions = []
    st.session_state.squat_results = None
    st.session_state.pose = None
    st.session_state.cap = None
    st.session_state.out_raw = None
    st.session_state.out_annot = None
    st.session_state.joint_angles_history = []
    st.session_state.frame_landmarks = []
    st.session_state.ai_analysis = None
    st.session_state.generated_image = None

    if keep_user:
        st.session_state.current_user = current_user
        st.session_state.users = users
    else:
        st.session_state.current_user = None
        st.session_state.users = {}

    st.session_state.user_session_id = None
    print("Session state reset complete")

# -----------------------------------------------------------------------------------
# 7) Initialize Global Session State
# -----------------------------------------------------------------------------------
print("Initializing global session state")
if "capture_running" not in st.session_state:
    st.session_state.capture_running = False
if "csv_file" not in st.session_state:
    st.session_state.csv_file = None
if "csv_writer" not in st.session_state:
    st.session_state.csv_writer = None
if "current_squat_count" not in st.session_state:
    st.session_state.current_squat_count = 0
if "squat_positions" not in st.session_state:
    st.session_state.squat_positions = []
if "squat_results" not in st.session_state:
    st.session_state.squat_results = None
if "pose" not in st.session_state:
    st.session_state.pose = None
if "cap" not in st.session_state:
    st.session_state.cap = None
if "out_raw" not in st.session_state:
    st.session_state.out_raw = None
if "out_annot" not in st.session_state:
    st.session_state.out_annot = None

if "current_user" not in st.session_state:
    st.session_state.current_user = None
if "users" not in st.session_state:
    st.session_state.users = {}
if "user_session_id" not in st.session_state:
    st.session_state.user_session_id = None
if "joint_angles_history" not in st.session_state:
    st.session_state.joint_angles_history = []
if "frame_landmarks" not in st.session_state:
    st.session_state.frame_landmarks = []
if "ai_analysis" not in st.session_state:
    st.session_state.ai_analysis = None
if "generated_image" not in st.session_state:
    st.session_state.generated_image = None

# Active tab management
if "active_tab" not in st.session_state:
    st.session_state.active_tab = None

print("Global session state initialization complete")

# -----------------------------------------------------------------------------------
# 8) Target Angles and Tolerance Settings
# -----------------------------------------------------------------------------------
TARGET_ANGLES = {
    'hip': 90.0,    # Target for hip angle in deep squat position (decreases from ~180° when standing)
    'knee': 90.0,   # Target for knee angle in deep squat position (decreases from ~180° when standing)
    'ankle': 25.0   # Target for ankle dorsiflexion angle (increases from ~10° when standing)
}
TOLERANCE = 5.0  # ±5°
print(f"Target angles set: Hip={TARGET_ANGLES['hip']}°, Knee={TARGET_ANGLES['knee']}°, Ankle={TARGET_ANGLES['ankle']}°, Tolerance={TOLERANCE}°")

# -----------------------------------------------------------------------------------
# 9) User Management Functions
# -----------------------------------------------------------------------------------
def create_user_folders(user_id):
    """Create folder structure for a user"""
    print(f"Creating user folders for: {user_id}")
    user_dir = os.path.join(USERS_DIR, user_id)
    
    folders = {
        "csv": os.path.join(user_dir, "csv"),
        "image": os.path.join(user_dir, "image"),
        "image_anno": os.path.join(user_dir, "image_anno"),
        "video": os.path.join(user_dir, "video"),
        "video_anno": os.path.join(user_dir, "video_anno"),
        "results": os.path.join(user_dir, "results"),
        "ai_images": os.path.join(user_dir, "ai_images")  # New folder for AI-generated images
    }
    
    for folder_name, folder_path in folders.items():
        os.makedirs(folder_path, exist_ok=True)
        print(f"  - Created {folder_name} folder: {folder_path}")
        
    return folders

def save_user_info(user_id, user_info):
    """Save user info to JSON file"""
    print(f"Saving user info for: {user_id}")
    user_dir = os.path.join(USERS_DIR, user_id)
    os.makedirs(user_dir, exist_ok=True)
    user_info_path = os.path.join(user_dir, "user_info.json")
    
    try:
        with open(user_info_path, 'w', encoding='utf-8') as f:
            json.dump(user_info, f, indent=4)
        print(f"User info saved to: {user_info_path}")
    except Exception as e:
        print(f"Error saving user info: {str(e)}")

def load_users():
    """Load all registered users' info"""
    print("Loading user information")
    users = {}
    if os.path.exists(USERS_DIR):
        user_dirs = [d for d in os.listdir(USERS_DIR) if os.path.isdir(os.path.join(USERS_DIR, d))]
        print(f"  - Found {len(user_dirs)} user directories")
        
        for user_id in user_dirs:
            user_dir = os.path.join(USERS_DIR, user_id)
            if os.path.isdir(user_dir):
                user_info_path = os.path.join(user_dir, "user_info.json")
                if os.path.exists(user_info_path):
                    try:
                        with open(user_info_path, 'r', encoding='utf-8') as f:
                            user_info = json.load(f)
                            users[user_id] = user_info
                            print(f"  - Loaded user: {user_id} ({user_info.get('name', 'Unknown')})")
                    except Exception as e:
                        print(f"  - Error loading user {user_id}: {str(e)}")
    
    print(f"User loading complete: {len(users)} users")
    return users

def create_session_id():
    """Generate a new session ID based on timestamp"""
    session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"New session ID created: {session_id}")
    return session_id

# -----------------------------------------------------------------------------------
# 10) OpenAI Functions
# -----------------------------------------------------------------------------------
def get_ai_analysis(angles, target_angles, tolerance):
    """Get AI analysis of squat posture using OpenAI GPT-4"""
    print("Starting AI analysis request")
    
    try:
        prompt = f"""
        You are a professional exercise posture analyst specializing in squat form. Analyze the user's squat posture based on the following joint angle measurements:
        
        Measured angles:
        - Hip angle: {angles['hip']:.1f}° (target: {target_angles['hip']}° ± {tolerance}°)
        - Knee angle: {angles['knee']:.1f}° (target: {target_angles['knee']}° ± {tolerance}°)
        - Ankle dorsiflexion angle: {angles['ankle']:.1f}° (target: {target_angles['ankle']}° ± {tolerance}°)
        
        Based on this data, please provide:
        1. A precise assessment of the user's posture (analysis by angle)
        2. What aspects are done well and what needs improvement
        3. 3-5 practical tips for posture correction
        4. 2-3 supplementary exercises that would help this user
        
        Please be professional, supportive, and detailed in your analysis.
        """
        
        print("Sending request to OpenAI API...")
        # Use the standard chat.completions API
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a professional exercise coach and squat posture analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        analysis = response.choices[0].message.content
        print("AI analysis complete")
        return analysis
        
    except Exception as e:
        print(f"AI analysis error: {str(e)}")
        return f"Error obtaining AI analysis: {str(e)}"

def generate_dalle_image(prompt):
    """Generate squat guidance image using DALL-E 3"""
    print(f"Starting DALL-E image generation: {prompt[:50]}...")
    
    try:
        st.info("Requesting image generation from OpenAI... this may take a moment.")
        print("Sending image generation request to OpenAI API...")
        
        # Using the current images.generate API
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            n=1,
            size="1024x1024",
            quality="standard"
        )
        
        image_url = response.data[0].url
        print(f"Image URL received: {image_url[:50]}...")
        
        image_response = requests.get(image_url, timeout=30)
        image = Image.open(BytesIO(image_response.content))
        print(f"Image downloaded: {image.size}")
        
        return image, image_url
    except Exception as e:
        st.error(f"Image generation error: {str(e)}")
        print(f"Image generation error: {str(e)}")
        
        if "RateLimitError" in str(e):
            st.error("Rate limit exceeded. Please try again later.")
        elif "AuthenticationError" in str(e) or "Authentication" in str(e):
            st.error("API key authentication failed. Please check your API key.")
        elif "timeout" in str(e).lower():
            st.error("Request timed out. Please try again later.")
        
        return None, None

# -----------------------------------------------------------------------------------
# 11) Core Analysis Functions
# -----------------------------------------------------------------------------------
def cleanup_resources():
    """Release all used resources"""
    print("Starting resource cleanup")
    try:
        if hasattr(st.session_state, 'out_raw') and st.session_state.out_raw:
            st.session_state.out_raw.release()
            st.session_state.out_raw = None
            print("  - Raw video writer released")

        if hasattr(st.session_state, 'out_annot') and st.session_state.out_annot:
            st.session_state.out_annot.release()
            st.session_state.out_annot = None
            print("  - Annotated video writer released")

        if hasattr(st.session_state, 'cap') and st.session_state.cap is not None:
            st.session_state.cap.release()
            st.session_state.cap = None
            print("  - Camera capture released")

        if hasattr(st.session_state, 'csv_file') and st.session_state.csv_file is not None:
            st.session_state.csv_file.close()
            st.session_state.csv_file = None
            print("  - CSV file closed")

        if hasattr(st.session_state, 'pose') and st.session_state.pose is not None:
            st.session_state.pose.close()
            st.session_state.pose = None
            print("  - MediaPipe Pose released")

        cv2.destroyAllWindows()
        print("Resource cleanup complete")
    except Exception as e:
        print(f"Error during resource cleanup: {str(e)}")
        print(traceback.format_exc())

def calculate_angle(a, b, c):
    """
    Calculate angle between three points (a, b, c)
    Returns angle in degrees
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
    return angle

def calculate_ankle_angle(ankle, knee):
    """
    발목의 배측굴곡 각도 계산 (90도에서 수평면-발목-무릎 각도를 뺀 값)
    """
    # 발목-무릎 벡터 계산
    ankle_knee_vector = [knee['x'] - ankle['x'], knee['y'] - ankle['y']]
    
    # 무릎이 발목보다 앞에 있는지 확인 (정상적인 스쿼트 자세)
    knee_in_front = ankle_knee_vector[0] > 0
    
    # 수평선 벡터 (발목에서 오른쪽으로)
    horizontal_vector = [1, 0]
    
    # 두 벡터 사이의 각도 계산
    dot_product = ankle_knee_vector[0] * horizontal_vector[0] + ankle_knee_vector[1] * horizontal_vector[1]
    magnitude_ankle_knee = (ankle_knee_vector[0]**2 + ankle_knee_vector[1]**2)**0.5
    magnitude_horizontal = 1
    
    # 안전하게 arccos 계산
    cos_value = max(-1, min(1, dot_product / (magnitude_ankle_knee * magnitude_horizontal)))
    angle_deg = np.degrees(np.arccos(cos_value))
    
    # 무릎의 위치에 따른 각도 조정
    if ankle_knee_vector[1] < 0:  # 무릎이 발목보다 위에 있음
        horizontal_ankle_angle = angle_deg
    else:  # 무릎이 발목보다 아래에 있음
        horizontal_ankle_angle = 180 - angle_deg
    
    # 배측굴곡 각도 계산
    dorsiflexion_angle = 90 - horizontal_ankle_angle
    
    # 무릎이 발목보다 뒤에 있는 비정상적인 자세라면 양수로 보정
    if not knee_in_front:
        dorsiflexion_angle = abs(dorsiflexion_angle)
    
    return dorsiflexion_angle

def calculate_side_angles(landmarks, side="right"):
    """Calculate angles for specified side of body"""
    angles = {}
    
    # Select landmarks based on side
    if side == "right":
        torso = landmarks[12]  # right shoulder
        hip = landmarks[24]    # right hip
        knee = landmarks[26]   # right knee
        ankle = landmarks[28]  # right ankle
    else:  # left side
        torso = landmarks[11]  # left shoulder
        hip = landmarks[23]    # left hip
        knee = landmarks[25]   # left knee
        ankle = landmarks[27]  # left ankle
    
    # Hip angle (torso-hip-knee) - this is measuring the front angle
    # When standing, this is ~180°, decreases as person squats
    hip_angle = calculate_angle(
        [torso['x'], torso['y']],
        [hip['x'], hip['y']],
        [knee['x'], knee['y']]
    )
    angles['hip'] = hip_angle
    
    # Knee angle (hip-knee-ankle) - this is measuring the back angle
    # When standing, this is ~180°, decreases as person squats
    knee_angle = calculate_angle(
        [hip['x'], hip['y']],
        [knee['x'], knee['y']],
        [ankle['x'], ankle['y']]
    )
    angles['knee'] = knee_angle
    
    # 새로운 발목 각도 계산 - 수평면과 발목-무릎 선 사이의 각도
    ankle_angle = calculate_ankle_angle(ankle, knee)
    angles['ankle'] = ankle_angle
    
    # Calculate visibility scores for each joint
    angles['visibility'] = {
        'hip': (hip['visibility'] + knee['visibility'] + torso['visibility']) / 3,
        'knee': (hip['visibility'] + knee['visibility'] + ankle['visibility']) / 3,
        'ankle': (knee['visibility'] + ankle['visibility']) / 2,  # 발가락 제외
        'overall': (hip['visibility'] + knee['visibility'] + ankle['visibility'] + torso['visibility']) / 4
    }
    
    return angles

def determine_best_angles(right_angles, left_angles, landmarks):
    """Determine which side's angles to use based on visibility, or average both sides."""
    final_angles = {}
    
    # Visibility threshold
    VISIBILITY_THRESHOLD = 0.7
    
    # Check overall visibility
    right_visible = right_angles['visibility']['overall'] > VISIBILITY_THRESHOLD
    left_visible = left_angles['visibility']['overall'] > VISIBILITY_THRESHOLD
    
    # Determine which angles to use for each joint
    for joint in ['hip', 'knee', 'ankle']:
        if right_visible and left_visible:
            # Both sides visible - average them
            final_angles[joint] = (right_angles[joint] + left_angles[joint]) / 2
        elif right_visible:
            # Only right side visible
            final_angles[joint] = right_angles[joint]
        elif left_visible:
            # Only left side visible
            final_angles[joint] = left_angles[joint]
        else:
            # Neither side has good visibility - use side with better visibility
            if right_angles['visibility'][joint] >= left_angles['visibility'][joint]:
                final_angles[joint] = right_angles[joint]
            else:
                final_angles[joint] = left_angles[joint]
    
    # Store which side was used for visualization
    if right_visible and left_visible:
        final_angles['side_used'] = 'both'
    elif right_visible:
        final_angles['side_used'] = 'right'
    elif left_visible:
        final_angles['side_used'] = 'left'
    else:
        # Determine which side had better overall visibility
        if right_angles['visibility']['overall'] >= left_angles['visibility']['overall']:
            final_angles['side_used'] = 'right'
        else:
            final_angles['side_used'] = 'left'
    
    return final_angles

def calculate_joint_angles(landmarks=None):
    """
    Calculate joint angles using both left and right sides of the body.
    Uses the side with better visibility, or averages both sides if both are visible.
    
    각 관절 각도의 설명:
    - Hip angle: 서 있을 때 약 180°, 스쿼트 시 감소 (torso-hip-knee)
    - Knee angle: 서 있을 때 약 180°, 스쿼트 시 감소 (hip-knee-ankle)
    - Ankle angle: 발목의 배측굴곡 각도, 서 있을 때 약 10°, 스쿼트 시 증가
    """
    if landmarks is None:
        print("Calculating average angles from stored squat positions...")
        # Calculate average angles from multiple squat positions
        angles = {
            'hip': [],
            'knee': [],
            'ankle': []
        }
        sides_used = []
        
        position_count = len(st.session_state.squat_positions)
        print(f"  - Positions to calculate: {position_count}")
        
        for pos_idx, position in enumerate(st.session_state.squat_positions):
            lms = position['landmarks']
            
            # Calculate angles for both sides
            right_angles = calculate_side_angles(lms, side="right")
            left_angles = calculate_side_angles(lms, side="left")
            
            # Determine which side to use or average both
            final_angles = determine_best_angles(right_angles, left_angles, lms)
            
            # Add to angle lists
            angles['hip'].append(final_angles['hip'])
            angles['knee'].append(final_angles['knee'])
            angles['ankle'].append(final_angles['ankle'])
            sides_used.append(final_angles.get('side_used', 'right'))
                
            if pos_idx % 5 == 0:  # Log only some positions
                print(f"  - Position {pos_idx+1}/{position_count} angles: Hip={angles['hip'][-1]:.1f}°, Knee={angles['knee'][-1]:.1f}°, Ankle={angles['ankle'][-1]:.1f}° (Using: {final_angles.get('side_used', 'right')})")

        # Calculate average angles
        avg_angles = {}
        for joint, values in angles.items():
            if values:
                avg_angles[joint] = sum(values) / len(values)
                print(f"  - {joint} average angle: {avg_angles[joint]:.2f}° (sample size: {len(values)})")
            else:
                avg_angles[joint] = 0
                print(f"  - {joint} angle calculation failed")
        
        # Determine most used side
        if sides_used:
            side_counts = {'right': 0, 'left': 0, 'both': 0}
            for side in sides_used:
                if side in side_counts:
                    side_counts[side] += 1
            
            most_used_side = max(side_counts, key=side_counts.get)
            avg_angles['side_used'] = most_used_side
            print(f"  - Most used side for calculations: {most_used_side} (counts: {side_counts})")
        
        return avg_angles
    
    else:
        # Calculate angles for specific frame for both sides
        right_angles = calculate_side_angles(landmarks, side="right")
        left_angles = calculate_side_angles(landmarks, side="left")
        
        # Determine which side to use or average both
        final_angles = determine_best_angles(right_angles, left_angles, landmarks)
        
        return final_angles

def load_logo():
    """로고 이미지를 로드하는 함수"""
    logo_dir = os.path.join(st.session_state.base_dir, "logo")
    print(f"로고 디렉토리 확인: {logo_dir}")
    
    # 로고 디렉토리에서 모든 이미지 파일 검색
    image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp']
    logo_files = []
    
    if os.path.exists(logo_dir):
        for file in os.listdir(logo_dir):
            file_lower = file.lower()
            if any(file_lower.endswith(ext) for ext in image_extensions):
                logo_files.append(os.path.join(logo_dir, file))
    
    print(f"발견된 로고 파일: {logo_files}")
    
    # 발견된 첫 번째 이미지 파일 사용
    if logo_files:
        try:
            logo_img = Image.open(logo_files[0])
            print(f"로고 이미지 로드됨: {logo_files[0]}")
            return logo_img
        except Exception as e:
            print(f"로고 로드 오류: {str(e)}")
            return None
    else:
        print("로고 파일을 찾을 수 없음")
        return None

def generate_angle_comparison_visualization(angles, target_angles):
    """Generate visualization comparing user's angles to target angles"""
    print("Generating angle comparison visualization...")
    # Data preparation
    categories = ['Hip Angle', 'Knee Angle', 'Ankle Angle']
    user_values = [angles['hip'], angles['knee'], angles['ankle']]
    target_values = [target_angles['hip'], target_angles['knee'], target_angles['ankle']]
    
    print(f"  - User angles: Hip={user_values[0]:.2f}°, Knee={user_values[1]:.2f}°, Ankle={user_values[2]:.2f}°")
    print(f"  - Target angles: Hip={target_values[0]}°, Knee={target_values[1]}°, Ankle={target_values[2]}°")
    
    # Create chart
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Bar positions and width
    x = np.arange(len(categories))
    width = 0.35
    
    # Create bars
    rects1 = ax.bar(x - width/2, user_values, width, label='Measured Angles', color='skyblue')
    rects2 = ax.bar(x + width/2, target_values, width, label='Target Angles', color='lightgreen')
    
    # Show differences
    for i in range(len(categories)):
        diff = user_values[i] - target_values[i]
        color = 'red' if abs(diff) > TOLERANCE else 'green'
        ax.annotate(f'{diff:+.1f}°', 
                   xy=(x[i], max(user_values[i], target_values[i]) + 5),
                   ha='center', va='bottom', 
                   color=color, fontweight='bold')
    
    # Decorate chart
    ax.set_ylabel('Angle (degrees)', fontsize=12)
    ax.set_title('User Joint Angles vs Target Angles', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    
    # Show tolerance range
    for i, target in enumerate(target_values):
        ax.axhspan(target - TOLERANCE, target + TOLERANCE, alpha=0.1, color='green', xmin=i/len(categories), xmax=(i+1)/len(categories))
    
    plt.tight_layout()
    print("Angle comparison visualization generated")
    return fig

def provide_clear_feedback(angles):
    """Generate clear feedback based on joint angles"""
    print("Generating posture feedback...")
    feedback = []
    detailed_feedback = []
    
    # Hip angle feedback - 목표 90도, 서 있을 때 ~180도에서 시작해 감소
    hip_diff = angles['hip'] - TARGET_ANGLES['hip']
    if abs(hip_diff) > TOLERANCE:
        if hip_diff > 0:  # 현재 각도가 목표보다 큼 (더 구부려야 함)
            feedback.append(f"Hip angle is {abs(hip_diff):.1f}° too large.")
            detailed_feedback.append("Squat deeper. Lower your hips more and bend your knees naturally toward the direction of your toes.")
            print(f"  - Hip feedback: Angle too large ({angles['hip']:.1f}° vs {TARGET_ANGLES['hip']}° target)")
        else:  # 현재 각도가 목표보다 작음 (덜 구부려야 함)
            feedback.append(f"Hip angle is {abs(hip_diff):.1f}° too small.")
            detailed_feedback.append("Don't squat too deep. Your thighs should be parallel to the floor.")
            print(f"  - Hip feedback: Angle too small ({angles['hip']:.1f}° vs {TARGET_ANGLES['hip']}° target)")
    else:
        print(f"  - Hip feedback: Within range ({angles['hip']:.1f}° vs {TARGET_ANGLES['hip']}° target)")
    
    # Knee angle feedback - 목표 90도, 서 있을 때 ~180도에서 시작해 감소
    knee_diff = angles['knee'] - TARGET_ANGLES['knee']
    if abs(knee_diff) > TOLERANCE:
        if knee_diff > 0:  # 현재 각도가 목표보다 큼 (더 구부려야 함)
            feedback.append(f"Knee angle is {abs(knee_diff):.1f}° too large.")
            detailed_feedback.append("Bend your knees more, ensuring they point in the same direction as your feet.")
            print(f"  - Knee feedback: Angle too large ({angles['knee']:.1f}° vs {TARGET_ANGLES['knee']}° target)")
        else:  # 현재 각도가 목표보다 작음 (덜 구부려야 함)
            feedback.append(f"Knee angle is {abs(knee_diff):.1f}° too small.")
            detailed_feedback.append("Your knees are bent too much. Reduce knee flexion and raise your upper body slightly.")
            print(f"  - Knee feedback: Angle too small ({angles['knee']:.1f}° vs {TARGET_ANGLES['knee']}° target)")
    else:
        print(f"  - Knee feedback: Within range ({angles['knee']:.1f}° vs {TARGET_ANGLES['knee']}° target)")
    
    # Ankle angle feedback - 목표 25도, 서 있을 때 ~10도 이하에서 시작해 증가
    ankle_diff = angles['ankle'] - TARGET_ANGLES['ankle']
    if abs(ankle_diff) > TOLERANCE:
        if ankle_diff > 0:  # 현재 각도가 목표보다 큼 (배측굴곡이 더 큼)
            feedback.append(f"Ankle dorsiflexion is {abs(ankle_diff):.1f}° too high.")
            detailed_feedback.append("Reduce ankle dorsiflexion. Try to maintain a more neutral ankle position.")
            print(f"  - Ankle feedback: Angle too high ({angles['ankle']:.1f}° vs {TARGET_ANGLES['ankle']}° target)")
        else:  # 현재 각도가 목표보다 작음 (배측굴곡이 더 작음)
            feedback.append(f"Ankle dorsiflexion is {abs(ankle_diff):.1f}° too low.")
            detailed_feedback.append("Increase ankle dorsiflexion. Allow knees to track forward more while keeping heels on the ground.")
            print(f"  - Ankle feedback: Angle too low ({angles['ankle']:.1f}° vs {TARGET_ANGLES['ankle']}° target)")
    else:
        print(f"  - Ankle feedback: Within range ({angles['ankle']:.1f}° vs {TARGET_ANGLES['ankle']}° target)")
    
    print(f"Feedback generation complete: {len(feedback)} issues found")
    return feedback, detailed_feedback

def update_angle_explanation():
    with st.expander("ℹ️ Angle Measurement Method and Meaning", expanded=False):
        st.write("""
        ### Angle Measurement Method and Meaning
        
        - **Hip angle**: 몸 앞쪽 각도로, 어깨-엉덩이-무릎 사이의 각도를 측정합니다. 
          * 서 있을 때는 약 180°에 가깝습니다
          * 스쿼트 시 엉덩이가 내려갈수록 각도가 감소합니다
          * 이상적인 딥 스쿼트 자세에서는 약 90°가 됩니다
        
        - **Knee angle**: 몸 뒤쪽 각도로, 엉덩이-무릎-발목 사이의 각도를 측정합니다.
          * 서 있을 때는 약 180°에 가깝습니다
          * 스쿼트 시 엉덩이가 내려갈수록 각도가 감소합니다
          * 이상적인 딥 스쿼트 자세에서는 약 90°가 됩니다
        
        - **Ankle angle**: 발목의 배측굴곡 각도로, 수평면과 발목-무릎 선 사이의 각도를 측정합니다.
          * 서 있을 때는 약 10° 이하입니다
          * 스쿼트 시 발목이 더 굽혀질수록 각도가 증가합니다
          * 이상적인 딥 스쿼트 자세에서는 약 25°가 됩니다
        
        각 관절의 각도가 목표 각도에 가까울수록 스쿼트 자세가 더 정확합니다. 이 값들은 XYZ 좌표를 기반으로 3D 공간에서 계산됩니다.
        """)

# -----------------------------------------------------------------------------------
# 12) Squat Capture Function
# -----------------------------------------------------------------------------------
def do_capture():
    """
    Capture squat movements through webcam
    - Ends after 5 squats or when "Stop Capture" is clicked
    - Includes 5-second countdown before starting
    - Saves raw video, annotated video, and CSV with landmarks
    """
    # Set active tab
    st.session_state.active_tab = "capture"
    
    print("Starting squat measurement")
    if not st.session_state.current_user:
        st.error("User must be registered first.")
        print("Capture aborted: No user registered")
        return
    
    # Create session ID
    st.session_state.user_session_id = create_session_id()
    
    # Create user folders
    user_folders = create_user_folders(st.session_state.current_user)
    
    # Create container
    capture_container = st.container()
    
    with capture_container:
        st.header("Squat Motion Capture")
        st.write("Perform 5 squats in front of the camera. Each squat will be automatically recorded.")
        
        # 5-second countdown
        st.subheader("Get Ready")
        st.write("Capture will start in 5 seconds.")
        countdown_placeholder = st.empty()
        for i in range(5, 0, -1):
            countdown_placeholder.write(f"### **{i}**")
            print(f"Countdown: {i}")
            time.sleep(1)
        countdown_placeholder.write("### **Start!**")
        time.sleep(1)
        countdown_placeholder.empty()

        # Clean up previous resources
        if st.session_state.cap is not None:
            st.session_state.cap.release()
            st.session_state.cap = None
        if st.session_state.pose is not None:
            st.session_state.pose.close()
            st.session_state.pose = None

        # Initialize session state
        st.session_state.capture_running = True
        st.session_state.current_squat_count = 0
        st.session_state.squat_positions = []
        st.session_state.joint_angles_history = []
        st.session_state.frame_landmarks = []

        # Create video and status display areas
        col1, col2 = st.columns([3, 1])
        
        with col1:
            video_placeholder = st.empty()
            st.subheader("Video Feed")
        
        with col2:
            status_text = st.empty()
            squat_count_text = st.empty()
            angles_text = st.empty()
            side_used_text = st.empty()  # New element to show which side is being used
            st.subheader("Status")
            stop_button_placeholder = st.empty()
            stop_button = stop_button_placeholder.button("Stop Capture", key="stop_capture", use_container_width=True)

        # Initialize MediaPipe Pose
        try:
            print("Initializing MediaPipe Pose...")
            st.session_state.pose = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            print("MediaPipe Pose initialized")
        except Exception as e:
            st.error(f"MediaPipe initialization error: {str(e)}")
            print(f"MediaPipe initialization failed: {str(e)}")
            st.session_state.capture_running = False
            return

        # Try different camera sources
        camera_sources = [0]
        if SYSTEM == 'Windows':
            camera_sources.extend([1, 2, 3, 4])
        else:
            camera_sources.extend(['0', '1'])
        
        print(f"Trying camera sources: {camera_sources}")
        success = False
        for source in camera_sources:
            try:
                print(f"Trying camera source {source}...")
                st.session_state.cap = cv2.VideoCapture(source)
                time.sleep(1)  # Wait for camera to initialize
                
                ret, frame = st.session_state.cap.read()
                if ret and frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0:
                    st.success(f"Camera connected successfully (source: {source})")
                    print(f"Camera connected (source: {source}, resolution: {frame.shape[1]}x{frame.shape[0]})")
                    success = True
                    break
                else:
                    print(f"Camera source {source} failed: Could not get valid frame")
                    st.session_state.cap.release()
                    st.session_state.cap = None
            except Exception as e:
                print(f"Camera source {source} open failed: {str(e)}")
                st.warning(f"Camera source {source} failed: {str(e)}")
        
        if not success:
            st.error("Could not open camera. Check connection and permissions.")
            print("All camera sources failed")
            st.session_state.capture_running = False
            return

        # Get camera properties
        width = int(st.session_state.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(st.session_state.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = st.session_state.cap.get(cv2.CAP_PROP_FPS) or 20.0
        print(f"Camera frame info: {width}x{height} @{fps:.1f}fps")

        # Test frame read
        test_ret, test_frame = st.session_state.cap.read()
        if not test_ret or test_frame is None:
            st.error("Cannot read first frame from camera.")
            print("First frame read failed")
            st.session_state.capture_running = False
            cleanup_resources()
            return
        else:
            video_placeholder.image(cv2.cvtColor(test_frame, cv2.COLOR_BGR2RGB), caption="Camera test image", use_column_width=True)
            print("Camera test image displayed")
            time.sleep(1)

        # Create output files
        try:
            print("Creating video and CSV files...")
            fourcc = get_video_codec()
            timestamp = st.session_state.user_session_id

            video_raw_path = os.path.join(user_folders["video"], f"squat_raw_{timestamp}.mp4")
            video_annot_path = os.path.join(user_folders["video_anno"], f"squat_annot_{timestamp}.mp4")
            print(f"Raw video path: {video_raw_path}")
            print(f"Annotated video path: {video_annot_path}")

            st.session_state.out_raw = cv2.VideoWriter(video_raw_path, fourcc, fps, (width, height))
            st.session_state.out_annot = cv2.VideoWriter(video_annot_path, fourcc, fps, (width, height))

            csv_path = os.path.join(user_folders["csv"], f"squat_landmarks_{timestamp}.csv")
            print(f"CSV file path: {csv_path}")
            st.session_state.csv_file = open(csv_path, "w", newline="", encoding="utf-8")
            st.session_state.csv_writer = csv.writer(st.session_state.csv_file)

            # Write CSV header
            header = ["frame", "timestamp", "squat_count"]
            for i in range(33):
                header += [f"lm_{i}_x", f"lm_{i}_y", f"lm_{i}_z", f"lm_{i}_visibility"]
            st.session_state.csv_writer.writerow(header)
            print("CSV header written")

            # Display file save info in an expandable section
            with st.expander("📁 File Save Information", expanded=False):
                st.write(f"**CSV File Path**: `{csv_path}`")
                st.write(f"**Raw Video Path**: `{video_raw_path}`")
                st.write(f"**Annotated Video Path**: `{video_annot_path}`")
        except Exception as e:
            st.error(f"File creation error: {str(e)}")
            print(f"File creation error: {str(e)}")
            cleanup_resources()
            st.session_state.capture_running = False
            return

        # Main capture loop
        is_squat_down = False
        frame_count = 0
        start_time = time.time()
        KNEE_ANGLE_THRESHOLD = 120  # Knee bend detection threshold
        print(f"Squat detection threshold set: knee angle < {KNEE_ANGLE_THRESHOLD}°")

        try:
            print("Starting squat capture loop")
            while st.session_state.capture_running and st.session_state.current_squat_count < 5 and not stop_button:
                ret, frame = st.session_state.cap.read()
                if not ret or frame is None:
                    status_text.warning("Camera frame read failed")
                    print("Frame read failed, retrying...")
                    time.sleep(0.5)
                    continue

                frame_count += 1
                current_time = time.time() - start_time

                raw_frame = frame.copy()
                frame = cv2.flip(frame, 1)  # Mirror horizontally
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                status_text.write(f"### Status: Measuring")
                squat_count_text.write(f"Squat Count: **{st.session_state.current_squat_count}/5**")
                
                if frame_count % 30 == 0:  # Log every 30 frames
                    print(f"Processing frame: {frame_count} (Squat count: {st.session_state.current_squat_count}/5)")

                # Process frame with MediaPipe
                results = st.session_state.pose.process(rgb)
                annotated_frame = frame.copy()

                if results.pose_landmarks:
                    # Get landmarks
                    landmarks = results.pose_landmarks.landmark
                    
                    # Save to CSV
                    row = [frame_count, current_time, st.session_state.current_squat_count]
                    for lm in landmarks:
                        row += [lm.x, lm.y, lm.z, lm.visibility]
                    st.session_state.csv_writer.writerow(row)

                    # Convert landmarks for easier processing
                    frame_landmarks = [{
                        'x': lm.x,
                        'y': lm.y,
                        'z': lm.z,
                        'visibility': lm.visibility
                    } for lm in landmarks]

                    # Calculate and store angles
                    frame_angles = calculate_joint_angles(frame_landmarks)
                    st.session_state.joint_angles_history.append({
                        'frame': frame_count,
                        'time': current_time,
                        'angles': frame_angles,
                        'squat_count': st.session_state.current_squat_count
                    })
                    st.session_state.frame_landmarks.append({
                        'frame': frame_count,
                        'time': current_time,
                        'landmarks': frame_landmarks,
                        'squat_count': st.session_state.current_squat_count
                    })

                    # Draw pose landmarks
                    mp_drawing.draw_landmarks(
                        annotated_frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                    )

                    # Get angles
                    knee_angle = frame_angles['knee']
                    hip_angle = frame_angles['hip']
                    ankle_angle = frame_angles['ankle']
                    side_used = frame_angles.get('side_used', 'right')  # Default to right if not specified

                    # Display angle info
                    angles_text.write(f"""
                    **Measured Angles**
                    - Hip: {hip_angle:.1f}°
                    - Knee: {knee_angle:.1f}°
                    - Ankle: {ankle_angle:.1f}°
                    """)
                    
                    # Display which side is being used
                    side_text = f"Using: {side_used.title()} Side"
                    if side_used == 'both':
                        side_text = "Using: Both Sides (Averaged)"
                    side_used_text.write(f"**{side_text}**")

                    # Add angle text to frame
                    cv2.putText(annotated_frame, f"Hip: {hip_angle:.1f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f"Knee: {knee_angle:.1f}", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f"Ankle: {ankle_angle:.1f}", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(annotated_frame, side_text, (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    # Squat detection
                    if not is_squat_down and knee_angle < KNEE_ANGLE_THRESHOLD:
                        is_squat_down = True
                        status_text.write(f"### Status: Squat down!")
                        print(f"Squat down detected! (Knee angle: {knee_angle:.1f}°)")
                    elif is_squat_down and knee_angle >= KNEE_ANGLE_THRESHOLD:
                        is_squat_down = False
                        st.session_state.current_squat_count += 1
                        status_text.write(f"### Status: {st.session_state.current_squat_count} squats completed!")
                        squat_count_text.write(f"Squat Count: **{st.session_state.current_squat_count}/5**")
                        print(f"Squat {st.session_state.current_squat_count} completed! (Knee angle: {knee_angle:.1f}°)")
                        
                        # Save squat position landmarks
                        st.session_state.squat_positions.append({'landmarks': frame_landmarks})
                        print(f"Squat position saved (total: {len(st.session_state.squat_positions)})")

                # Write frames to video
                st.session_state.out_raw.write(raw_frame)
                st.session_state.out_annot.write(annotated_frame)

                # Save frame images periodically
                if frame_count % 10 == 0:  # Every 10 frames
                    img_filename_raw = os.path.join(user_folders["image"], f"frame_{timestamp}_{frame_count:04d}.jpg")
                    img_filename_annot = os.path.join(user_folders["image_anno"], f"frame_{timestamp}_{frame_count:04d}.jpg")
                    cv2.imwrite(img_filename_raw, raw_frame)
                    cv2.imwrite(img_filename_annot, annotated_frame)
                    if frame_count % 30 == 0:  # Log every 30 frames
                        print(f"Frame images saved: {frame_count:04d} (raw & annotated)")

                # Display the frame
                video_placeholder.image(annotated_frame, channels="BGR", use_column_width=True)
                stop_button = stop_button_placeholder.button("Stop Capture", key=f"stop_capture_{frame_count}", use_container_width=True)
                time.sleep(0.01)

        except Exception as e:
            st.error(f"Error during capture: {str(e)}")
            print(f"Capture error: {str(e)}")
            print(traceback.format_exc())
        finally:
            stop_button_placeholder.empty()
            cleanup_resources()

            if st.session_state.current_squat_count >= 5:
                st.success("5 squats completed! Click 'Evaluate Squat Posture' to see the results.")
                print("5 squats completed")
                
                # Save angle history
                angles_history_path = os.path.join(user_folders["results"], f"angles_history_{timestamp}.csv")
                angles_df = pd.DataFrame([
                    {
                        'frame': item['frame'],
                        'time': item['time'],
                        'hip_angle': item['angles']['hip'],
                        'knee_angle': item['angles']['knee'],
                        'ankle_angle': item['angles']['ankle'],
                        'side_used': item['angles'].get('side_used', 'right'),  # Add side_used information
                        'squat_count': item['squat_count']
                    }
                    for item in st.session_state.joint_angles_history
                ])
                angles_df.to_csv(angles_history_path, index=False)
                print(f"Angle history saved: {angles_history_path} ({len(angles_df)} rows)")
                
                with st.expander("📊 Saved Data Information", expanded=False):
                    st.write(f"**Angle History Saved**: `{angles_history_path}`")
                    st.dataframe(angles_df.head())
                
                # Button to go to evaluation page
                if st.button("Go to Evaluation", key="goto_evaluation"):
                    st.session_state.active_tab = "evaluate"
                    st.rerun()
            else:
                st.info(f"Squat measurement stopped. (Current count: {st.session_state.current_squat_count})")
                print(f"Squat stopped: {st.session_state.current_squat_count} squats")

# -----------------------------------------------------------------------------------
# 13) Squat Evaluation Function (Enhanced with OpenAI)
# -----------------------------------------------------------------------------------
def evaluate_squat():
    """Evaluate squat posture with AI-powered analysis"""
    # Set active tab
    st.session_state.active_tab = "evaluate"
    
    print("Starting squat posture evaluation with AI")
    
    # Create evaluation container
    eval_container = st.container()
    
    with eval_container:
        st.header("Squat Posture Evaluation Results")
        
        if not st.session_state.squat_positions:
            st.warning("Please perform squat measurement first.")
            print("Evaluation aborted: No squat position data")
            return

        if not st.session_state.current_user:
            st.warning("User registration required.")
            print("Evaluation aborted: No user registered")
            return

        user_folders = create_user_folders(st.session_state.current_user)
        timestamp = st.session_state.user_session_id or create_session_id()
        print(f"Evaluation timestamp: {timestamp}, positions: {len(st.session_state.squat_positions)}")

        # Calculate joint angles
        angles = calculate_joint_angles()
        st.session_state.squat_results = angles

        # Display angle analysis
        st.subheader("Joint Angle Analysis Results (Average)")
        data = []
        joint_names = {
            'hip': 'Hip Angle',
            'knee': 'Knee Angle',
            'ankle': 'Ankle Dorsiflexion'
        }

        for joint in ['hip', 'knee', 'ankle']:
            measured_angle = angles[joint]
            target_angle = TARGET_ANGLES[joint]
            diff = round(measured_angle - target_angle, 1)
            data.append([
                joint_names[joint],
                f"{target_angle}°",
                f"{measured_angle:.1f}°",
                f"{diff:+.1f}°"
            ])

        df = pd.DataFrame(data, columns=["Metric", "Target", "Measured", "Difference"])
        st.table(df)
        
        # Show which side was used for measurements
        if 'side_used' in angles:
            side_text = f"Using measurements from: {angles['side_used'].title()} Side"
            if angles['side_used'] == 'both':
                side_text = "Using measurements from: Both Sides (Averaged)"
            st.info(side_text)

        # Save results
        result_csv_path = os.path.join(user_folders["results"], f"squat_results_{timestamp}.csv")
        df.to_csv(result_csv_path, index=False)
        print(f"Evaluation results saved: {result_csv_path}")
        with st.expander("📁 Result File Information", expanded=False):
            st.write(f"Results saved to: `{result_csv_path}`")

        # Explanation of angles
        update_angle_explanation()

        # Angle comparison visualization
        st.subheader("Angle Comparison Visualization")
        comparison_fig = generate_angle_comparison_visualization(angles, TARGET_ANGLES)
        st.pyplot(comparison_fig)
        comparison_fig_path = os.path.join(user_folders["results"], f"angle_comparison_{timestamp}.png")
        comparison_fig.savefig(comparison_fig_path, dpi=150, bbox_inches='tight')
        print(f"Angle comparison visualization saved: {comparison_fig_path}")

        # AI Analysis Section
        st.subheader("💡 AI Posture Analysis")
        
        if st.session_state.ai_analysis is None:
            with st.spinner("AI is analyzing your squat posture... this may take a moment."):
                st.session_state.ai_analysis = get_ai_analysis(angles, TARGET_ANGLES, TOLERANCE)
                
                # Save the AI analysis
                ai_analysis_path = os.path.join(user_folders["results"], f"ai_analysis_{timestamp}.txt")
                with open(ai_analysis_path, 'w', encoding='utf-8') as f:
                    f.write(st.session_state.ai_analysis)
                print(f"AI analysis saved to: {ai_analysis_path}")
        
        # Display AI analysis with nicer formatting
        st.markdown(st.session_state.ai_analysis)

        # Basic feedback
        st.subheader("Quick Feedback Summary")
        feedback, detailed_feedback = provide_clear_feedback(angles)
        
        if feedback:
            for i, (issue, detail) in enumerate(zip(feedback, detailed_feedback)):
                with st.expander(f"📌 {issue}", expanded=True):
                    st.info(f"{detail}")
        else:
            st.success("All joint angles are within the target range. Excellent squat posture!")

        # Angle change graph
        if st.session_state.joint_angles_history:
            print(f"Generating angle change graph (data points: {len(st.session_state.joint_angles_history)})")
            st.subheader("Joint Angle Changes During Squat Exercise")
            df_angles = pd.DataFrame([
                {
                    'time': item['time'],
                    'hip_angle': item['angles']['hip'],
                    'knee_angle': item['angles']['knee'],
                    'ankle_angle': item['angles']['ankle'],
                    'side_used': item['angles'].get('side_used', 'right'),  # Add side used information
                    'squat_count': item['squat_count']
                }
                for item in st.session_state.joint_angles_history
            ])
            
            # Data summary statistics
            print(f"Angle data summary:")
            print(f"  - Time range: {df_angles['time'].min():.1f}s ~ {df_angles['time'].max():.1f}s")
            print(f"  - Hip angle range: {df_angles['hip_angle'].min():.1f}° ~ {df_angles['hip_angle'].max():.1f}°")
            print(f"  - Knee angle range: {df_angles['knee_angle'].min():.1f}° ~ {df_angles['knee_angle'].max():.1f}°")
            print(f"  - Ankle angle range: {df_angles['ankle_angle'].min():.1f}° ~ {df_angles['ankle_angle'].max():.1f}°")
            
            # Show side distribution
            side_counts = df_angles['side_used'].value_counts()
            print(f"  - Side used distribution: {side_counts.to_dict()}")
            
            # 각도 변화 그래프 생성 부분 - 타겟 라인이 겹치지 않도록 수정
            fig, ax = plt.subplots(figsize=(12, 7))

            # 측정된 각도 (실선)
            ax.plot(df_angles['time'], df_angles['hip_angle'], label='Hip Angle', color='royalblue', linewidth=2)
            ax.plot(df_angles['time'], df_angles['knee_angle'], label='Knee Angle', color='darkorange', linewidth=2)
            ax.plot(df_angles['time'], df_angles['ankle_angle'], label='Ankle Dorsiflexion', color='forestgreen', linewidth=2)
            
            # 타겟 라인 (각각 다른 스타일로 표시)
            # 힙 각도와 무릎 각도가 모두 90도라서 겹치므로 오프셋 적용
            ax.axhline(y=TARGET_ANGLES['hip']-1, color='red', linestyle='--', linewidth=3, alpha=0.8, label='Target Hip Angle')
            ax.axhline(y=TARGET_ANGLES['knee']+1, color='green', linestyle='-.', linewidth=3, alpha=0.8, label='Target Knee Angle')
            ax.axhline(y=TARGET_ANGLES['ankle'], color='blue', linestyle=':', linewidth=3, alpha=0.8, label='Target Ankle Angle')
            
            # Mark squat transitions
            squat_changes = df_angles.loc[df_angles['squat_count'].diff() != 0]
            for idx, row in squat_changes.iterrows():
                ax.axvline(x=row['time'], color='gray', linestyle='-', alpha=0.3)
                ax.text(row['time'], ax.get_ylim()[1]*0.95, f"Squat {int(row['squat_count'])}", 
                        rotation=90, verticalalignment='top')
            
            # 그래프 설정
            ax.set_xlabel('Time (seconds)', fontsize=14)
            ax.set_ylabel('Angle (degrees)', fontsize=14)
            ax.set_title('Joint Angle Changes During Squat Exercise', fontsize=16)
            ax.legend(fontsize=12, loc='upper right')
            ax.grid(True, alpha=0.3)
            
            # Save and display
            angles_plot_path = os.path.join(user_folders["results"], f"angles_plot_{timestamp}.png")
            plt.savefig(angles_plot_path, dpi=150, bbox_inches='tight')
            print(f"Angle change graph saved: {angles_plot_path}")
            st.pyplot(fig)
            
            # Show side distribution in a pie chart
            st.subheader("Measurement Side Distribution")
            fig_pie, ax_pie = plt.subplots(figsize=(6, 6))
            side_counts.plot.pie(autopct='%1.1f%%', ax=ax_pie, title='Body Side Used for Measurements')
            pie_path = os.path.join(user_folders["results"], f"side_distribution_{timestamp}.png")
            plt.savefig(pie_path, dpi=150, bbox_inches='tight')
            st.pyplot(fig_pie)

        # Information on squat measurement principles
        with st.expander("📚 Squat Measurement Principles", expanded=False):
            st.write("""
            ## Squat Measurement Principles

            This system uses the MediaPipe Pose model to analyze your squat posture in real-time.

            1. **Joint Coordinate Extraction**: 33 body landmarks are recognized in real-time through the camera.
            2. **Angle Calculation**: Angles of major joints (hip, knee, ankle) are calculated using trigonometry.
            3. **Target Angle Comparison**: The differences between your measured angles and ideal squat posture angles are analyzed.
            4. **Posture Evaluation**: Evaluates whether measured angles are within the target range and suggests improvements.
            5. **AI Analysis**: Uses artificial intelligence to provide detailed professional feedback on your form.

            Collected data is converted into graphs and visual materials to provide intuitive feedback.
            """)

        # Joint angle distribution histograms
        st.subheader("Joint Angle Distribution Histograms")
        print("Generating angle distribution histograms")
        bins = 20
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Create histograms
        sns.histplot(df_angles['hip_angle'], kde=True, bins=bins, ax=axes[0])
        axes[0].axvline(TARGET_ANGLES['hip'], color='r', linestyle='--')
        axes[0].set_title('Hip Angle Distribution')
        
        sns.histplot(df_angles['knee_angle'], kde=True, bins=bins, ax=axes[1])
        axes[1].axvline(TARGET_ANGLES['knee'], color='r', linestyle='--')
        axes[1].set_title('Knee Angle Distribution')
        
        sns.histplot(df_angles['ankle_angle'], kde=True, bins=bins, ax=axes[2])
        axes[2].axvline(TARGET_ANGLES['ankle'], color='r', linestyle='--')
        axes[2].set_title('Ankle Dorsiflexion Distribution')
        
        plt.tight_layout()
        
        # Save and display
        heatmap_path = os.path.join(user_folders["results"], f"angle_heatmap_{timestamp}.png")
        plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
        print(f"Angle distribution histograms saved: {heatmap_path}")
        st.pyplot(fig)

        # Calculate scores
        hip_score = max(0, 10 - abs(angles['hip'] - TARGET_ANGLES['hip'])) / 10 * 100
        knee_score = max(0, 10 - abs(angles['knee'] - TARGET_ANGLES['knee'])) / 10 * 100
        ankle_score = max(0, 10 - abs(angles['ankle'] - TARGET_ANGLES['ankle'])) / 10 * 100
        overall_score = (hip_score * 0.4) + (knee_score * 0.4) + (ankle_score * 0.2)
        print(f"Score calculation: Hip={hip_score:.1f}, Knee={knee_score:.1f}, Ankle={ankle_score:.1f}, Overall={overall_score:.1f}")

        # Display scores
        st.subheader("Overall Evaluation Score")
        score_col1, score_col2, score_col3, score_col4 = st.columns(4)
        with score_col1:
            st.metric("Hip Score", f"{hip_score:.1f}")
        with score_col2:
            st.metric("Knee Score", f"{knee_score:.1f}")
        with score_col3:
            st.metric("Ankle Score", f"{ankle_score:.1f}")
        with score_col4:
            st.metric("Overall Score", f"{overall_score:.1f}")
        
        # Save scores
        scores_df = pd.DataFrame({
            'hip_score': [hip_score],
            'knee_score': [knee_score],
            'ankle_score': [ankle_score],
            'overall_score': [overall_score]
        })
        scores_df.to_csv(os.path.join(user_folders["results"], f"scores_{timestamp}.csv"), index=False)
        print(f"Scores saved: {os.path.join(user_folders['results'], f'scores_{timestamp}.csv')}")

        # Create radar chart
        st.subheader("Strengths/Weaknesses Analysis (Radar Chart)")
        print("Generating radar chart...")
        categories = ['Hip', 'Knee', 'Ankle']
        scores = [hip_score/100, knee_score/100, ankle_score/100]

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, polar=True)
        
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        scores += scores[:1]
        angles += angles[:1]
        categories += categories[:1]
        
        ax.plot(angles, scores, 'o-', linewidth=2, label='User Score')
        ax.fill(angles, scores, alpha=0.25)
        
        ideal_scores = [1.0] * len(categories)
        ax.plot(angles, ideal_scores, 'o-', linewidth=2, label='Ideal Score')
        ax.fill(angles, ideal_scores, alpha=0.1)
        
        ax.set_thetagrids(np.degrees(angles[:-1]), categories[:-1])
        ax.set_ylim(0, 1)
        ax.grid(True)
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        # Save and display
        radar_chart_path = os.path.join(user_folders["results"], f"radar_chart_{timestamp}.png")
        plt.savefig(radar_chart_path, dpi=150, bbox_inches='tight')
        print(f"Radar chart saved: {radar_chart_path}")
        st.pyplot(fig)

        # Final info and navigation button
        st.write("Evaluation complete! Now you can check the 'Custom Guide' tab for personalized squat guidance with AI-generated visuals.")
        
        # Button to go to custom guide page
        if st.button("Go to Custom Guide", key="goto_guide"):
            st.session_state.active_tab = "guide"
            st.rerun()

# -----------------------------------------------------------------------------------
# 14) Custom Guide Function with DALL-E Image Generation
# -----------------------------------------------------------------------------------
def generate_squat_guide():
    """Generate personalized squat guide with AI analysis and DALL-E visualization"""
    # Set active tab
    st.session_state.active_tab = "guide"
    
    print("Starting personalized squat guide generation with AI")
    
    # Create guide container
    guide_container = st.container()
    
    with guide_container:
        st.header("AI-Powered Personalized Squat Guide")
        
        if not st.session_state.squat_results:
            st.warning("No squat evaluation data available. Please complete 'Squat Posture Evaluation' first.")
            print("Guide aborted: No squat result data")
            return

        angles = st.session_state.squat_results
        user_id = st.session_state.current_user
        user_name = st.session_state.users[user_id]['name']
        user_folders = create_user_folders(user_id)
        timestamp = st.session_state.user_session_id or create_session_id()
        print(f"Generating AI squat guide for {user_name}")
        
        # Show which body side was used for the analysis
        if 'side_used' in angles:
            side_text = f"Analysis based on: {angles['side_used'].title()} Side"
            if angles['side_used'] == 'both':
                side_text = "Analysis based on: Both Sides (Averaged)"
            st.info(side_text)

        # Show AI analysis from evaluation if available
        if st.session_state.ai_analysis:
            st.subheader(f"🧠 AI Analysis for {user_name}")
            st.markdown(st.session_state.ai_analysis)
        
        # Generate personalized visual guide with DALL-E
        st.subheader("🖼️ Personalized Squat Visualization")
        
        if st.session_state.generated_image is None:
            # Identify main issue
            issues = []
            if abs(angles['hip'] - TARGET_ANGLES['hip']) > TOLERANCE:
                diff = angles['hip'] - TARGET_ANGLES['hip']
                if diff > 0:
                    issues.append("Hip angle is too large - needs to squat deeper")
                else:
                    issues.append("Hip angle is too small - squatting too deep")
            
            if abs(angles['knee'] - TARGET_ANGLES['knee']) > TOLERANCE:
                diff = angles['knee'] - TARGET_ANGLES['knee']
                if diff > 0:
                    issues.append("Knee angle is too large - needs to bend knees more")
                else:
                    issues.append("Knee angle is too small - knees bent too much")
            
            if abs(angles['ankle'] - TARGET_ANGLES['ankle']) > TOLERANCE:
                diff = angles['ankle'] - TARGET_ANGLES['ankle']
                if diff > 0:
                    issues.append("Ankle dorsiflexion is too high - reduce ankle flexion")
                else:
                    issues.append("Ankle dorsiflexion is too low - increase ankle flexion")

            print(f"Identified issues: {len(issues)}")
            
            # Prepare DALL-E prompt based on analysis
            if issues:
                main_issue = issues[0]
                image_prompt = f"""
                Create a detailed instructional image showing the correct squat posture, focusing on fixing this issue:
                "{main_issue}"
                
                The image should:
                - Show a clear side view of proper squat form
                - Highlight the correct alignment for hip angle (90°), knee angle (90°), and ankle dorsiflexion (25°)
                - Include clear anatomical labels and directional arrows showing proper movement
                - Use professional visual style with clean background
                - Include text annotations explaining key form points
                
                Make it suitable for a fitness instruction guide.
                """
            else:
                image_prompt = """
                Create a detailed instructional image showing perfect squat posture with proper form.
                
                The image should:
                - Show a clear side view of ideal squat form with 90° hip angle, 90° knee angle, and proper ankle dorsiflexion (25°)
                - Highlight the correct alignment of spine, knees tracking over toes, and weight distribution
                - Include professional anatomical labels and directional indicators
                - Use clean background with clear visibility of the technique
                - Include text annotations explaining 3-4 key form points
                
                Make it suitable for a fitness instruction guide.
                """
            
            # Generate image with DALL-E
            with st.spinner("Generating personalized squat guide image... this may take a moment."):
                image, image_url = generate_dalle_image(image_prompt)
                if image:
                    st.session_state.generated_image = image
                    image_path = os.path.join(user_folders["ai_images"], f"squat_guide_image_{timestamp}.png")
                    image.save(image_path)
                    print(f"Squat guide image saved: {image_path}")
        
        # Display the generated image
        if st.session_state.generated_image:
            st.image(st.session_state.generated_image, caption="AI-Generated Personalized Squat Guide", use_column_width=True)
        else:
            st.warning("⚠️ Image generation failed. Check your OpenAI API key and connection.")
            print("Image generation failed")
            
            # Show text-based guidance instead
            st.subheader("📋 Text-Based Guidance")
            st.markdown("""
            Since the image generation failed, here's a text-based guide instead:
            
            1. **Foot Position**: Stand with feet shoulder-width apart, toes pointed slightly outward (15-30°)
            2. **Hip Hinge**: Begin the movement by pushing your hips back as if sitting in a chair
            3. **Knee Alignment**: Keep knees tracking over toes, not collapsing inward
            4. **Depth**: Lower until thighs are parallel to the ground (hip and knee at approximately 90°)
            5. **Back Position**: Maintain a neutral spine throughout the movement
            6. **Weight Distribution**: Keep weight centered over mid-foot, not shifting to toes
            7. **Ankle Mobility**: Allow appropriate ankle dorsiflexion (about 25°) while keeping heels on the ground
            """)
        
        # Key points
        st.subheader("💡 Key Points for Improvement")
        
        # Identify areas for improvement
        hip_diff = angles['hip'] - TARGET_ANGLES['hip']
        knee_diff = angles['knee'] - TARGET_ANGLES['knee']
        ankle_diff = angles['ankle'] - TARGET_ANGLES['ankle']
        
        # Create columns for layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Hip Position")
            if abs(hip_diff) > TOLERANCE:
                if hip_diff > 0:
                    st.markdown("🔽 **Lower your hips more**")
                    st.markdown("- Push your hips back and down")
                    st.markdown("- Focus on 'sitting' deeper into the squat")
                    st.markdown("- Aim for thighs parallel to the ground")
                else:
                    st.markdown("🔼 **Don't squat too deep**")
                    st.markdown("- Control your descent")
                    st.markdown("- Stop when thighs are parallel to ground")
                    st.markdown("- Focus on stability at the bottom position")
            else:
                st.markdown("✅ **Excellent hip angle!**")
                st.markdown("- Maintain this good hip positioning")
                st.markdown("- Focus on consistency in depth")
            
            st.markdown("#### Ankle Position")
            if abs(ankle_diff) > TOLERANCE:
                if ankle_diff > 0:
                    st.markdown("📉 **Reduce ankle dorsiflexion**")
                    st.markdown("- Try to maintain a more neutral ankle position")
                    st.markdown("- Distribute weight more toward mid-foot")
                    st.markdown("- Don't let knees track too far forward")
                else:
                    st.markdown("📈 **Increase ankle dorsiflexion**")
                    st.markdown("- Allow knees to track further forward")
                    st.markdown("- Keep heels planted firmly on the ground")
                    st.markdown("- Work on ankle mobility exercises")
            else:
                st.markdown("✅ **Great ankle positioning!**")
                st.markdown("- Continue with good ankle mobility")
                st.markdown("- Maintain weight distribution")
        
        with col2:
            st.markdown("#### Knee Position")
            if abs(knee_diff) > TOLERANCE:
                if knee_diff > 0:
                    st.markdown("🔽 **Bend knees more**")
                    st.markdown("- Deepen knee flexion")
                    st.markdown("- Ensure knees track over toes")
                    st.markdown("- Keep knees aligned with feet")
                else:
                    st.markdown("🔼 **Reduce knee bend**")
                    st.markdown("- Push hips back more")
                    st.markdown("- Don't let knees travel too far forward")
                    st.markdown("- Focus on hip-dominant movement")
            else:
                st.markdown("✅ **Perfect knee angle!**")
                st.markdown("- Maintain this ideal knee positioning")
                st.markdown("- Continue tracking knees with toes")
            
            st.markdown("#### Overall Form")
            if abs(hip_diff) <= TOLERANCE and abs(knee_diff) <= TOLERANCE and abs(ankle_diff) <= TOLERANCE:
                st.markdown("🏆 **Your squat form is excellent!**")
                st.markdown("- Focus on progressive overload")
                st.markdown("- Consider adding weight or resistance")
                st.markdown("- Maintain this technical proficiency")
            else:
                st.markdown("🔄 **Practice for improvement**")
                st.markdown("- Film yourself from the side")
                st.markdown("- Start with bodyweight before adding load")
                st.markdown("- Consider working with a coach")
        
        # Supplementary exercises section
        st.subheader("💪 Recommended Supplementary Exercises")
        
        # Based on identified issues, recommend specific exercises
        with st.expander("View Recommended Exercises", expanded=True):
            if abs(hip_diff) > TOLERANCE:
                st.markdown("#### Hip Mobility & Strength")
                st.markdown("""
                1. **Goblet Squats**: Hold a kettlebell or dumbbell close to chest, focus on depth and form
                2. **Hip Bridges**: Strengthen glutes and teach proper hip extension
                3. **Deep Squat Hold**: Practice holding at bottom position to improve mobility
                """)
            
            if abs(knee_diff) > TOLERANCE:
                st.markdown("#### Knee Stability & Strength")
                st.markdown("""
                1. **Bulgarian Split Squats**: Improve single-leg stability and knee alignment
                2. **Wall Sits**: Build isometric strength in the quadriceps
                3. **Step-Ups**: Strengthen knee stabilizers and improve control
                """)
            
            if abs(ankle_diff) > TOLERANCE:
                st.markdown("#### Ankle Mobility & Stability")
                st.markdown("""
                1. **Downward-Facing Dog to Runner's Lunge**: Dynamic ankle mobility exercise
                2. **Banded Ankle Mobilization**: Use a resistance band to improve ankle dorsiflexion
                3. **Calf Raises & Knee-to-Wall Stretch**: Improve ankle flexibility and strength
                """)
            
            if abs(hip_diff) <= TOLERANCE and abs(knee_diff) <= TOLERANCE and abs(ankle_diff) <= TOLERANCE:
                st.markdown("#### Advanced Progression")
                st.markdown("""
                1. **Front Squats**: Challenge core stability while maintaining good squat mechanics
                2. **Single-Leg Squats**: Improve balance, coordination, and address asymmetries
                3. **Pause Squats**: Add isometric holds at the bottom to build strength and control
                """)
        
        # Follow-up recommendations
        with st.expander("📋 Next Steps", expanded=True):
            st.markdown("""
            ### Follow-up Recommendations
            
            1. **Re-assessment**: Measure your squat form again after 2 weeks of practice
            2. **Progressive Training**: Follow this sequence:
               - Master bodyweight form first
               - Add controlled tempo (3 second descent, pause, controlled ascent)
               - Gradually add resistance as form improves
            3. **Mobility Work**: Spend 5-10 minutes daily on mobility for hips, knees, and ankles
            4. **Video Analysis**: Record yourself regularly from multiple angles to self-assess
            5. **Consistency**: Practice proper squats 2-3 times weekly for best progress
            """)
            
        print("AI squat guide generation complete")

# -----------------------------------------------------------------------------------
# 15) Main Application Layout
# -----------------------------------------------------------------------------------
def main():
    """Main application layout and interaction logic"""
    print("Loading user information...")
    st.session_state.users = load_users()
    
    # Declare global variables at the beginning of the function
    global OPENAI_API_KEY, client
    
    # Center-aligned logo
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        logo_img = load_logo()
        if logo_img:
            st.image(logo_img, width=300, use_column_width=True)
        else:
            st.info("로고 이미지를 찾을 수 없습니다. 로고 폴더를 확인하세요.")

    st.title("AI Squat Analysis")
    st.markdown("---")

    # API Key input in sidebar
    with st.sidebar:
        st.subheader("OpenAI API Settings")
        api_key = st.text_input("Enter OpenAI API Key", value=OPENAI_API_KEY, type="password")
        if api_key != OPENAI_API_KEY:
            OPENAI_API_KEY = api_key
            client = OpenAI(api_key=OPENAI_API_KEY)
            st.success("API key updated")
        
        st.markdown("---")
        st.markdown("#### About")
        st.markdown("""
        This application uses:
        - Computer Vision (MediaPipe) for pose tracking
        - OpenAI GPT-4 for intelligent posture analysis
        - DALL-E 3 for personalized guide images
        
        Ensure you have a valid OpenAI API key with access to GPT-4 and DALL-E 3.
        """)


    # Create tabs
    tabs = ["User Management", "Squat Measurement", "Posture Evaluation", "Custom Guide"]
    active_tab = st.session_state.active_tab if "active_tab" in st.session_state else "user"
    
    # Calculate tab index
    tab_index = 0
    if active_tab == "capture":
        tab_index = 1
    elif active_tab == "evaluate":
        tab_index = 2
    elif active_tab == "guide":
        tab_index = 3
    
    tab1, tab2, tab3, tab4 = st.tabs(tabs)
    
    # User Management Tab
    with tab1:
        st.header("User Management")
        
        user_col1, user_col2 = st.columns(2)

        with user_col1:
            st.subheader("Register New User")
            new_user_name = st.text_input("Name")
            new_user_age = st.number_input("Age", min_value=1, max_value=120, value=25)
            new_user_gender = st.selectbox("Gender", ["Male", "Female"])
            
            if st.button("Register User", use_container_width=True):
                if new_user_name:
                    user_id = f"user_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
                    user_info = {
                        "name": new_user_name,
                        "age": new_user_age,
                        "gender": new_user_gender,
                        "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    save_user_info(user_id, user_info)
                    create_user_folders(user_id)
                    st.session_state.users[user_id] = user_info
                    st.session_state.current_user = user_id
                    st.success(f"✅ {new_user_name} has been registered!")
                    print(f"New user registered: {new_user_name} (ID: {user_id})")
                else:
                    st.error("❌ Please enter a name.")
                    print("User registration failed: No name entered")

        with user_col2:
            st.subheader("Select Existing User")
            if st.session_state.users:
                user_options = {f"{info['name']} ({user_id})": user_id 
                                for user_id, info in st.session_state.users.items()}
                selected_user_display = st.selectbox("User List", list(user_options.keys()))
                
                if st.button("Select", use_container_width=True):
                    selected_user_id = user_options[selected_user_display]
                    st.session_state.current_user = selected_user_id
                    # Reset AI analysis and image when switching users
                    st.session_state.ai_analysis = None
                    st.session_state.generated_image = None
                    st.success(f"✅ Selected user: {st.session_state.users[selected_user_id]['name']}")
                    print(f"User selected: {st.session_state.users[selected_user_id]['name']} (ID: {selected_user_id})")
            else:
                st.info("📝 No registered users. Please register a user first.")
                print("No registered users")

        # Display current user info
        st.markdown("---")
        if st.session_state.current_user:
            user_info = st.session_state.users[st.session_state.current_user]
            st.subheader("Current User Information")
            
            info_col1, info_col2, info_col3 = st.columns(3)
            with info_col1:
                st.metric("Name", user_info['name'])
            with info_col2:
                st.metric("Age", user_info['age'])
            with info_col3:
                st.metric("Gender", user_info['gender'])
                
            st.info(f"📋 User folder: {os.path.join(USERS_DIR, st.session_state.current_user)}")
        else:
            st.warning("⚠️ Please select or register a user.")

    # Squat Measurement Tab
    with tab2:
        st.header("Squat Measurement")
        
        if not st.session_state.current_user:
            st.warning("⚠️ Please register or select a user first.")
        else:
            st.markdown("""
            This feature measures your squat movement in real-time through the webcam. Click the Start button and perform 5 squats.
            The camera will detect your squat posture and analyze the angles of each joint in real-time.
            """)
            
            if st.button("📸 Start Squat Measurement", use_container_width=True):
                print(f"Starting squat measurement (user: {st.session_state.users[st.session_state.current_user]['name']})")
                do_capture()
                
            # Display measurement status and results
            if st.session_state.current_squat_count > 0:
                st.success(f"✅ {st.session_state.current_squat_count} squats measured")
                if st.button("📊 Evaluate Squat Posture with AI", use_container_width=True):
                    st.session_state.active_tab = "evaluate"
                    st.rerun()
            
            with st.expander("ℹ️ Measurement Instructions", expanded=False):
                st.markdown("""
                ### How to Measure
                1. Click the Start button and prepare in front of the camera.
                2. After a 5-second countdown, measurement will begin.
                3. Position yourself 2-3m from the camera so your full body is visible.
                4. Perform 5 squats at a comfortable pace.
                5. Once complete, you can proceed to AI-powered posture evaluation.
                
                ### Tips
                - Side view provides more accurate analysis.
                - Wear clothing that allows joints to be visible and movement to be unrestricted.
                - Place feet at shoulder width with toes pointed slightly outward.
                """)

    # Posture Evaluation Tab
    with tab3:
        st.header("AI Squat Posture Evaluation")
        
        if not st.session_state.current_user:
            st.warning("⚠️ Please register or select a user first.")
        elif st.session_state.current_squat_count == 0:
            st.warning("⚠️ Please complete squat measurement first.")
        else:
            st.markdown("""
            Evaluate your squat posture with AI analysis. Your joint angles will be compared with ideal squat form angles, 
            and our AI will provide detailed professional feedback.
            """)
            
            if st.button("🔍 Start AI Posture Evaluation", use_container_width=True):
                # Reset AI analysis to ensure fresh analysis
                st.session_state.ai_analysis = None
                print(f"Starting squat evaluation (squats: {st.session_state.current_squat_count})")
                evaluate_squat()
                
            # If evaluation results exist, show guide button
            if st.session_state.squat_results:
                if st.button("🧠 View AI-Generated Guide", use_container_width=True):
                    st.session_state.active_tab = "guide"
                    st.rerun()
            
            with st.expander("📈 AI Evaluation Features", expanded=False):
                st.markdown("""
                ### AI-Powered Squat Analysis
                
                Our system combines computer vision with advanced AI to analyze your squat form:
                
                1. **Precise Joint Angle Measurement**: Standard angles are calculated between key joints.
                
                2. **GPT-4 Analysis**: Professional AI analysis of your form with specific corrections.
                
                3. **Visual Comparisons**: Charts showing your angles compared to ideal form.
                
                4. **Progression Tracking**: Monitor improvements over time as you practice correct form.
                
                5. **Personalized Feedback**: Custom advice based on your specific body mechanics.
                """)

    # Custom Guide Tab
    with tab4:
        st.header("AI-Generated Custom Squat Guide")
        
        if not st.session_state.current_user:
            st.warning("⚠️ Please register or select a user first.")
        elif not st.session_state.squat_results:
            st.warning("⚠️ Please complete squat evaluation first.")
        else:
            st.markdown("""
            Based on your evaluation results, our AI will create a personalized squat guide, including a custom visualization
            of proper form tailored to your specific needs.
            """)
            
            if st.button("🧠 Generate AI Squat Guide & Visualization", use_container_width=True):
                print(f"Generating AI squat guide (user: {st.session_state.users[st.session_state.current_user]['name']})")
                generate_squat_guide()
                
            with st.expander("ℹ️ About AI Guide Features", expanded=False):
                st.markdown("""
                ### AI-Powered Personalized Guidance
                
                This guide combines multiple AI technologies to provide:
                
                1. **Custom Analysis**: GPT-4 analyzes your specific form issues and provides targeted advice.
                
                2. **Visual Learning**: DALL-E 3 creates a personalized instructional image showing proper form.
                
                3. **Targeted Exercises**: Specific supplementary exercises to address your unique needs.
                
                4. **Progressive Plan**: Clear steps to improve your form over time.
                
                5. **Professional Expertise**: Guidance based on professional coaching principles.
                """)

# Run the main application
if __name__ == "__main__":
    main()