import streamlit as st
import os
import sys
import platform
import datetime
import time
import traceback
import json
import csv
from io import BytesIO
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import requests
import numpy as np
from pathlib import Path
import importlib

# 환경 변수 관리 - dotenv 라이브러리가 있으면 로드
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("Environment variables loaded from .env file")
except ImportError:
    print("dotenv not installed, using environment variables directly")

# 클라우드 환경 확인
is_cloud_env = (
    os.environ.get('IS_STREAMLIT_CLOUD') == 'True' or 
    'STREAMLIT_SHARING_MODE' in os.environ or 
    'DYNO' in os.environ or
    os.environ.get('CLOUD_ENV') == 'True'
)

# 디버깅 정보 출력
print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")
print(f"Platform: {sys.platform}")
print(f"Running in cloud environment: {is_cloud_env}")

# 조건부로 시각화 라이브러리 임포트
try:
    import cv2
    import mediapipe as mp
    opencv_available = True
    print("OpenCV and MediaPipe imported successfully")
    # Mediapipe Setup
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    mp_drawing_styles = mp.solutions.drawing_styles
except ImportError as e:
    opencv_available = False
    print(f"OpenCV/MediaPipe import error: {e}")
    # 호환성을 위한 더미 객체 생성
    class DummyClass:
        def __getattr__(self, name):
            return lambda *args, **kwargs: None
    
    class DummySolutions:
        def __init__(self):
            self.drawing_utils = DummyClass()
            self.pose = DummyClass()
            self.drawing_styles = DummyClass()
    
    class DummyVideoCapture:
        def __init__(self, *args, **kwargs):
            pass
        
        def release(self):
            pass
            
        def read(self):
            return False, None
            
        def get(self, *args):
            return 0
    
    cv2 = DummyClass()
    cv2.VideoWriter_fourcc = lambda *args: 0
    cv2.VideoWriter = DummyClass
    cv2.VideoCapture = DummyVideoCapture
    cv2.cvtColor = lambda *args, **kwargs: None
    cv2.putText = lambda *args, **kwargs: None
    cv2.imwrite = lambda *args, **kwargs: None
    cv2.destroyAllWindows = lambda: None
    
    mp = DummySolutions()
    mp_drawing = mp.drawing_utils
    mp_pose = mp.pose
    mp_drawing_styles = mp.drawing_styles

# OpenAI API 설정
try:
    # 최신 방식의 OpenAI 클라이언트 임포트 시도
    from openai import OpenAI
    openai_available = True
    openai_new_client = True
    print("OpenAI client (new version) imported successfully")
except ImportError:
    try:
        # 이전 버전 OpenAI 모듈 임포트 시도
        import openai
        openai_available = True
        openai_new_client = False
        print("OpenAI module (old version) imported successfully")
    except ImportError:
        openai_available = False
        openai_new_client = False
        print("OpenAI library import error")
        # 더미 OpenAI 클라이언트 생성
        class DummyOpenAI:
            def __init__(self, api_key=None):
                self.api_key = api_key
                self.chat = DummyClass()
                self.chat.completions = DummyClass()
                self.images = DummyClass()
                self.images.generate = DummyClass()
        
        class DummyCompletion:
            @staticmethod
            def create(*args, **kwargs):
                class DummyChoices:
                    class DummyMessage:
                        content = "OpenAI API not available. Please provide a valid API key."
                    
                    class DummyChoice:
                        def __init__(self):
                            self.message = DummyChoices.DummyMessage()
                    
                    choices = [DummyChoice()]
                
                return DummyChoices()
                
        class DummyImage:
            @staticmethod
            def create(*args, **kwargs):
                return {"data": [{"url": ""}]}
        
        if not openai_new_client:
            openai = DummyClass()
            openai.Completion = DummyCompletion
            openai.ChatCompletion = DummyCompletion
            openai.Image = DummyImage
        
        OpenAI = DummyOpenAI

# 임시 파일 저장을 위한 방향
def get_temp_dir():
    """환경에 따라 적절한 임시 디렉토리 반환"""
    if is_cloud_env:
        # 클라우드 환경에서는 /tmp 사용
        temp_dir = "/tmp/healthnai_app"
    else:
        # 로컬에서는 현재 디렉토리 아래 temp 폴더 사용
        temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")
    
    # 디렉토리가 없으면 생성
    os.makedirs(temp_dir, exist_ok=True)
    return temp_dir

# -----------------------------------------------------------------------------------
# 1) 경로 설정 - 크로스 플랫폼 호환
# -----------------------------------------------------------------------------------
# 운영체제 감지
SYSTEM = platform.system()  # 'Windows', 'Darwin'(Mac), 'Linux'
print(f"Operating system: {SYSTEM}")

# 기본 디렉토리 설정
DEFAULT_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 클라우드 환경에서의 디렉토리 처리 향상
if is_cloud_env:
    # Streamlit Cloud 환경에서는 임시 디렉토리 사용
    DEFAULT_BASE_DIR = get_temp_dir()
    print(f"Cloud environment detected, using temp directory: {DEFAULT_BASE_DIR}")

# Streamlit 세션 상태에 기본 디렉토리 설정
if 'base_dir' not in st.session_state:
    st.session_state.base_dir = DEFAULT_BASE_DIR

# 디버그 출력
print(f"Base directory set to: {st.session_state.base_dir}")

# 필요한 폴더 경로 정의
DATA_DIR = os.path.join(st.session_state.base_dir, "data")
USERS_DIR = os.path.join(DATA_DIR, "users")
STANDARD_IMG_DIR = os.path.join(DATA_DIR, "imagestandard")
STATIC_DIR = os.path.join(DEFAULT_BASE_DIR, "static")
LOGO_PATH = os.path.join(STATIC_DIR, "logo", "healthnai_logo.png")

print(f"DATA_DIR: {DATA_DIR}")
print(f"USERS_DIR: {USERS_DIR}")
print(f"STANDARD_IMG_DIR: {STANDARD_IMG_DIR}")
print(f"STATIC_DIR: {STATIC_DIR}")
print(f"LOGO_PATH: {LOGO_PATH}")

# 디렉토리 생성
os.makedirs(st.session_state.base_dir, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(USERS_DIR, exist_ok=True)
os.makedirs(STANDARD_IMG_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOGO_PATH), exist_ok=True)

# 경로 확인
required_paths = [
    st.session_state.base_dir,
    DATA_DIR,
    USERS_DIR,
    STANDARD_IMG_DIR,
    STATIC_DIR,
    os.path.dirname(LOGO_PATH)
]
for path in required_paths:
    if not os.path.exists(path):
        print(f"Creating path: {path}")
        os.makedirs(path, exist_ok=True)
    else:
        print(f"Path verified: {path}")

# 샘플 로고 생성 (없는 경우)
if not os.path.exists(LOGO_PATH) and is_cloud_env:
    try:
        # 간단한 샘플 로고 생성
        sample_logo = Image.new('RGB', (300, 100), color=(73, 109, 137))
        
        # 실행 중 PIL 모듈이 없는 경우 대비
        try:
            from PIL import ImageDraw
            draw = ImageDraw.Draw(sample_logo)
            draw.text((100, 40), "HealthnAI", fill=(255, 255, 255))
        except ImportError:
            pass
            
        os.makedirs(os.path.dirname(LOGO_PATH), exist_ok=True)
        sample_logo.save(LOGO_PATH)
        print(f"Sample logo created at: {LOGO_PATH}")
    except Exception as e:
        print(f"Failed to create sample logo: {e}")

# -----------------------------------------------------------------------------------
# 2) OpenAI API 설정
# -----------------------------------------------------------------------------------
# 환경 변수에서 API 키 가져오기
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
print(f"OpenAI API key set: {'Yes' if OPENAI_API_KEY else 'No'}")

# OpenAI 클라이언트 초기화
if openai_available:
    if openai_new_client:
        client = OpenAI(api_key=OPENAI_API_KEY)
        print("Initialized OpenAI client (new version)")
    else:
        openai.api_key = OPENAI_API_KEY
        client = None
        print("Initialized OpenAI module (old version)")
else:
    client = OpenAI(api_key=OPENAI_API_KEY)
    print("Using dummy OpenAI client")

# -----------------------------------------------------------------------------------
# 4) 페이지 설정
# -----------------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Squat Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 스타일링
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
# 5) 비디오 코덱 설정 - 플랫폼별 지원
# -----------------------------------------------------------------------------------
def get_video_codec():
    """플랫폼에 적합한 비디오 코덱 반환"""
    if not opencv_available:
        return 0  # OpenCV를 사용할 수 없는 경우
    
    if SYSTEM == 'Windows':
        # Windows 코덱 옵션
        codecs_to_try = ['XVID', 'MJPG', 'H264', 'X264', 'WMV1']
        for codec in codecs_to_try:
            try:
                codec_value = cv2.VideoWriter_fourcc(*codec)
                print(f"Windows: {codec} codec success ({codec_value})")
                return codec_value
            except Exception as e:
                print(f"Windows: {codec} codec failed: {str(e)}")
                continue
        # 폴백
        print("Windows: Using fallback MJPG codec")
        return cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    else:  # Mac/Linux
        try:
            codec_value = cv2.VideoWriter_fourcc(*'mp4v')  # macOS 권장
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
# 6) 세션 상태 초기화 함수
# -----------------------------------------------------------------------------------
def reset_session_state(keep_user=True):
    """세션 상태 변수 초기화 (선택적으로 사용자 데이터 유지)"""
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
# 7) 전역 세션 상태 초기화
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

# 활성 탭 관리
if "active_tab" not in st.session_state:
    st.session_state.active_tab = None

print("Global session state initialization complete")

# -----------------------------------------------------------------------------------
# 8) 목표 각도 및 허용 오차 설정
# -----------------------------------------------------------------------------------
TARGET_ANGLES = {
    'hip': 90.0,    # 깊은 스쿼트 자세에서의 엉덩이 각도 목표 (서 있을 때 ~180°에서 감소)
    'knee': 90.0,   # 깊은 스쿼트 자세에서의 무릎 각도 목표 (서 있을 때 ~180°에서 감소)
    'ankle': 25.0   # 발목 배측굴곡 각도 목표 (서 있을 때 ~10°에서 증가)
}
TOLERANCE = 5.0  # ±5°
print(f"Target angles set: Hip={TARGET_ANGLES['hip']}°, Knee={TARGET_ANGLES['knee']}°, Ankle={TARGET_ANGLES['ankle']}°, Tolerance={TOLERANCE}°")

# -----------------------------------------------------------------------------------
# 9) 사용자 관리 함수
# -----------------------------------------------------------------------------------
def create_user_folders(user_id):
    """사용자 폴더 구조 생성"""
    print(f"Creating user folders for: {user_id}")
    user_dir = os.path.join(USERS_DIR, user_id)
    
    folders = {
        "csv": os.path.join(user_dir, "csv"),
        "image": os.path.join(user_dir, "image"),
        "image_anno": os.path.join(user_dir, "image_anno"),
        "video": os.path.join(user_dir, "video"),
        "video_anno": os.path.join(user_dir, "video_anno"),
        "results": os.path.join(user_dir, "results"),
        "ai_images": os.path.join(user_dir, "ai_images")  # AI 생성 이미지용 새 폴더
    }
    
    for folder_name, folder_path in folders.items():
        os.makedirs(folder_path, exist_ok=True)
        print(f"  - Created {folder_name} folder: {folder_path}")
        
    return folders

def save_user_info(user_id, user_info):
    """사용자 정보를 JSON 파일로 저장"""
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
    """등록된 모든 사용자 정보 로드"""
    print("Loading user information")
    users = {}
    if os.path.exists(USERS_DIR):
        try:
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
        except Exception as e:
            print(f"Error listing users directory: {str(e)}")
    
    print(f"User loading complete: {len(users)} users")
    return users

def create_session_id():
    """타임스탬프 기반 새 세션 ID 생성"""
    session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"New session ID created: {session_id}")
    return session_id

# -----------------------------------------------------------------------------------
# 10) OpenAI 함수
# -----------------------------------------------------------------------------------
def get_ai_analysis(angles, target_angles, tolerance):
    """OpenAI GPT-4를 사용한 스쿼트 자세 AI 분석"""
    print("Starting AI analysis request")
    
    if not openai_available or not OPENAI_API_KEY:
        return "OpenAI API key not provided or OpenAI library not available. Please enter your API key in the sidebar."
    
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
        
        if openai_new_client:
            # 새 OpenAI 클라이언트 API 사용
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
        else:
            # 이전 OpenAI API 사용
            response = openai.ChatCompletion.create(
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
    """DALL-E 3을 사용한 스쿼트 가이드 이미지 생성"""
    print(f"Starting DALL-E image generation: {prompt[:50]}...")
    
    if not openai_available or not OPENAI_API_KEY:
        return None, None
    
    try:
        st.info("Requesting image generation from OpenAI... this may take a moment.")
        print("Sending image generation request to OpenAI API...")
        
        if openai_new_client:
            # 새 OpenAI 클라이언트 API 사용
            response = client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                n=1,
                size="1024x1024",
                quality="standard"
            )
            image_url = response.data[0].url
        else:
            # 이전 OpenAI API 사용
            response = openai.Image.create(
                prompt=prompt,
                n=1,
                size="1024x1024"
            )
            image_url = response["data"][0]["url"]
        
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
# 11) 핵심 분석 함수
# -----------------------------------------------------------------------------------
def cleanup_resources():
    """사용된 모든 리소스 해제"""
    if not opencv_available:
        return
        
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
    세 점(a, b, c) 사이의 각도 계산
    각도를 도(degrees)로 반환
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
    """지정된 신체 측면의 각도 계산"""
    angles = {}
    
    # 측면에 따른 랜드마크 선택
    if side == "right":
        torso = landmarks[12]  # 오른쪽 어깨
        hip = landmarks[24]    # 오른쪽 엉덩이
        knee = landmarks[26]   # 오른쪽 무릎
        ankle = landmarks[28]  # 오른쪽 발목
    else:  # 왼쪽
        torso = landmarks[11]  # 왼쪽 어깨
        hip = landmarks[23]    # 왼쪽 엉덩이
        knee = landmarks[25]   # 왼쪽 무릎
        ankle = landmarks[27]  # 왼쪽 발목
    
    # 엉덩이 각도 (몸통-엉덩이-무릎) - 앞쪽 각도 측정
    # 서 있을 때 ~180°, 스쿼트 시 감소
    hip_angle = calculate_angle(
        [torso['x'], torso['y']],
        [hip['x'], hip['y']],
        [knee['x'], knee['y']]
    )
    angles['hip'] = hip_angle
    
    # 무릎 각도 (엉덩이-무릎-발목) - 뒤쪽 각도 측정
    # 서 있을 때 ~180°, 스쿼트 시 감소
    knee_angle = calculate_angle(
        [hip['x'], hip['y']],
        [knee['x'], knee['y']],
        [ankle['x'], ankle['y']]
    )
    angles['knee'] = knee_angle
    
    # 새로운 발목 각도 계산 - 수평면과 발목-무릎 선 사이의 각도
    ankle_angle = calculate_ankle_angle(ankle, knee)
    angles['ankle'] = ankle_angle
    
    # 각 관절의 가시성 점수 계산
    angles['visibility'] = {
        'hip': (hip['visibility'] + knee['visibility'] + torso['visibility']) / 3,
        'knee': (hip['visibility'] + knee['visibility'] + ankle['visibility']) / 3,
        'ankle': (knee['visibility'] + ankle['visibility']) / 2,  # 발가락 제외
        'overall': (hip['visibility'] + knee['visibility'] + ankle['visibility'] + torso['visibility']) / 4
    }
    
    return angles

def determine_best_angles(right_angles, left_angles, landmarks):
    """가시성에 따라 어떤 측면의 각도를 사용할지 결정하거나, 양쪽 모두 평균."""
    final_angles = {}
    
    # 가시성 임계값
    VISIBILITY_THRESHOLD = 0.7
    
    # 전체 가시성 확인
    right_visible = right_angles['visibility']['overall'] > VISIBILITY_THRESHOLD
    left_visible = left_angles['visibility']['overall'] > VISIBILITY_THRESHOLD
    
    # 각 관절에 대해 사용할 각도 결정
    for joint in ['hip', 'knee', 'ankle']:
        if right_visible and left_visible:
            # 양쪽 모두 가시적 - 평균화
            final_angles[joint] = (right_angles[joint] + left_angles[joint]) / 2
        elif right_visible:
            # 오른쪽만 가시적
            final_angles[joint] = right_angles[joint]
        elif left_visible:
            # 왼쪽만 가시적
            final_angles[joint] = left_angles[joint]
        else:
            # 양쪽 모두 가시성이 좋지 않음 - 더 나은 가시성의 쪽 사용
            if right_angles['visibility'][joint] >= left_angles['visibility'][joint]:
                final_angles[joint] = right_angles[joint]
            else:
                final_angles[joint] = left_angles[joint]
    
    # 시각화를 위해 어떤 측면이 사용되었는지 저장
    if right_visible and left_visible:
        final_angles['side_used'] = 'both'
    elif right_visible:
        final_angles['side_used'] = 'right'
    elif left_visible:
        final_angles['side_used'] = 'left'
    else:
        # 전체적으로 더 나은 가시성을 가진 측면 결정
        if right_angles['visibility']['overall'] >= left_angles['visibility']['overall']:
            final_angles['side_used'] = 'right'
        else:
            final_angles['side_used'] = 'left'
    
    return final_angles

def calculate_joint_angles(landmarks=None):
    """
    신체의 양쪽을 모두 사용하여 관절 각도 계산.
    더 나은 가시성을 가진 쪽을 사용하거나, 양쪽 모두 가시적이면 평균화.
    
    각 관절 각도의 설명:
    - Hip angle: 서 있을 때 약 180°, 스쿼트 시 감소 (torso-hip-knee)
    - Knee angle: 서 있을 때 약 180°, 스쿼트 시 감소 (hip-knee-ankle)
    - Ankle angle: 발목의 배측굴곡 각도, 서 있을 때 약 10°, 스쿼트 시 증가
    """
    if landmarks is None:
        print("Calculating average angles from stored squat positions...")
        # 여러 스쿼트 위치에서 평균 각도 계산
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
            
            # 양쪽의 각도 계산
            right_angles = calculate_side_angles(lms, side="right")
            left_angles = calculate_side_angles(lms, side="left")
            
            # 어떤 쪽을 사용할지 결정하거나 양쪽 평균
            final_angles = determine_best_angles(right_angles, left_angles, lms)
            
            # 각도 목록에 추가
            angles['hip'].append(final_angles['hip'])
            angles['knee'].append(final_angles['knee'])
            angles['ankle'].append(final_angles['ankle'])
            sides_used.append(final_angles.get('side_used', 'right'))
                
            if pos_idx % 5 == 0:  # 일부 위치만 로그
                print(f"  - Position {pos_idx+1}/{position_count} angles: Hip={angles['hip'][-1]:.1f}°, Knee={angles['knee'][-1]:.1f}°, Ankle={angles['ankle'][-1]:.1f}° (Using: {final_angles.get('side_used', 'right')})")

        # 평균 각도 계산
        avg_angles = {}
        for joint, values in angles.items():
            if values:
                avg_angles[joint] = sum(values) / len(values)
                print(f"  - {joint} average angle: {avg_angles[joint]:.2f}° (sample size: {len(values)})")
            else:
                avg_angles[joint] = 0
                print(f"  - {joint} angle calculation failed")
        
        # 가장 많이 사용된 측면 결정
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
        # 양쪽 모두에 대해 특정 프레임의 각도 계산
        right_angles = calculate_side_angles(landmarks, side="right")
        left_angles = calculate_side_angles(landmarks, side="left")
        
        # 어떤 쪽을 사용할지 결정하거나 양쪽 평균
        final_angles = determine_best_angles(right_angles, left_angles, landmarks)
        
        return final_angles

def load_logo():
    """로고 이미지를 로드하는 함수"""
    logo_dir = os.path.join(STATIC_DIR, "logo")
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
        # 샘플 로고 생성
        try:
            sample_logo = Image.new('RGB', (300, 100), color=(73, 109, 137))
            
            # 실행 중 PIL 모듈이 없는 경우 대비
            try:
                from PIL import ImageDraw
                from PIL import ImageFont
                draw = ImageDraw.Draw(sample_logo)
                
                # 기본 폰트 사용
                draw.text((100, 40), "HealthnAI", fill=(255, 255, 255))
            except ImportError:
                pass
                
            # 로고 디렉토리 생성
            os.makedirs(logo_dir, exist_ok=True)
            
            # 샘플 로고 저장
            sample_logo_path = os.path.join(logo_dir, "healthnai_logo.png")
            sample_logo.save(sample_logo_path)
            print(f"샘플 로고 생성: {sample_logo_path}")
            
            return sample_logo
        except Exception as e:
            print(f"샘플 로고 생성 오류: {str(e)}")
            return None

def generate_angle_comparison_visualization(angles, target_angles):
    """사용자 각도와 목표 각도를 비교하는 시각화 생성"""
    print("Generating angle comparison visualization...")
    # 데이터 준비
    categories = ['Hip Angle', 'Knee Angle', 'Ankle Angle']
    user_values = [angles['hip'], angles['knee'], angles['ankle']]
    target_values = [target_angles['hip'], target_angles['knee'], target_angles['ankle']]
    
    print(f"  - User angles: Hip={user_values[0]:.2f}°, Knee={user_values[1]:.2f}°, Ankle={user_values[2]:.2f}°")
    print(f"  - Target angles: Hip={target_values[0]}°, Knee={target_values[1]}°, Ankle={target_values[2]}°")
    
    # 차트 생성
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 막대 위치 및 너비
    x = np.arange(len(categories))
    width = 0.35
    
    # 막대 생성
    rects1 = ax.bar(x - width/2, user_values, width, label='Measured Angles', color='skyblue')
    rects2 = ax.bar(x + width/2, target_values, width, label='Target Angles', color='lightgreen')
    
    # 차이 표시
    for i in range(len(categories)):
        diff = user_values[i] - target_values[i]
        color = 'red' if abs(diff) > TOLERANCE else 'green'
        ax.annotate(f'{diff:+.1f}°', 
                   xy=(x[i], max(user_values[i], target_values[i]) + 5),
                   ha='center', va='bottom', 
                   color=color, fontweight='bold')
    
    # 차트 장식
    ax.set_ylabel('Angle (degrees)', fontsize=12)
    ax.set_title('User Joint Angles vs Target Angles', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    
    # 허용 오차 범위 표시
    for i, target in enumerate(target_values):
        ax.axhspan(target - TOLERANCE, target + TOLERANCE, alpha=0.1, color='green', xmin=i/len(categories), xmax=(i+1)/len(categories))
    
    plt.tight_layout()
    print("Angle comparison visualization generated")
    return fig

def provide_clear_feedback(angles):
    """관절 각도를 기반으로 명확한 피드백 생성"""
    print("Generating posture feedback...")
    feedback = []
    detailed_feedback = []
    
    # 엉덩이 각도 피드백 - 목표 90도, 서 있을 때 ~180도에서 시작해 감소
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
    
    # 무릎 각도 피드백 - 목표 90도, 서 있을 때 ~180도에서 시작해 감소
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
    
    # 발목 각도 피드백 - 목표 25도, 서 있을 때 ~10도 이하에서 시작해 증가
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
# 12) 스쿼트 캡처 함수
# -----------------------------------------------------------------------------------
def do_capture():
    """
    웹캠을 통한 스쿼트 동작 캡처
    - 5회 스쿼트 후 또는 "캡처 중지" 클릭 시 종료
    - 시작 전 5초 카운트다운 포함
    - 원본 비디오, 주석 처리된 비디오, 랜드마크 CSV 저장
    """
    if is_cloud_env or not opencv_available:
        st.error("This feature requires OpenCV and a webcam, which are not available in the cloud environment.")
        st.info("Please run this application locally for full functionality.")
        return
        
    # 활성 탭 설정
    st.session_state.active_tab = "capture"
    
    print("Starting squat measurement")
    if not st.session_state.current_user:
        st.error("User must be registered first.")
        print("Capture aborted: No user registered")
        return
    
    # 세션 ID 생성
    st.session_state.user_session_id = create_session_id()
    
    # 사용자 폴더 생성
    user_folders = create_user_folders(st.session_state.current_user)
    
    # 컨테이너 생성
    capture_container = st.container()
    
    with capture_container:
        st.header("Squat Motion Capture")
        st.write("Perform 5 squats in front of the camera. Each squat will be automatically recorded.")
        
        # 5초 카운트다운
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

        # 이전 리소스 정리
        if st.session_state.cap is not None:
            st.session_state.cap.release()
            st.session_state.cap = None
        if st.session_state.pose is not None:
            st.session_state.pose.close()
            st.session_state.pose = None

        # 세션 상태 초기화
        st.session_state.capture_running = True
        st.session_state.current_squat_count = 0
        st.session_state.squat_positions = []
        st.session_state.joint_angles_history = []
        st.session_state.frame_landmarks = []

        # 비디오 및 상태 표시 영역 생성
        col1, col2 = st.columns([3, 1])
        
        with col1:
            video_placeholder = st.empty()
            st.subheader("Video Feed")
        
        with col2:
            status_text = st.empty()
            squat_count_text = st.empty()
            angles_text = st.empty()
            side_used_text = st.empty()  # 어떤 측면이 사용되고 있는지 표시하는 새 요소
            st.subheader("Status")
            stop_button_placeholder = st.empty()
            stop_button = stop_button_placeholder.button("Stop Capture", key="stop_capture", use_container_width=True)

        # MediaPipe Pose 초기화
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

        # 다양한 카메라 소스 시도
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
                time.sleep(1)  # 카메라 초기화 대기
                
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

        # 카메라 속성 가져오기
        width = int(st.session_state.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(st.session_state.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = st.session_state.cap.get(cv2.CAP_PROP_FPS) or 20.0
        print(f"Camera frame info: {width}x{height} @{fps:.1f}fps")

        # 프레임 읽기 테스트
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

        # 출력 파일 생성
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

            # CSV 헤더 작성
# CSV 헤더 작성
            header = ["frame", "timestamp", "squat_count"]
            for i in range(33):
                header += [f"lm_{i}_x", f"lm_{i}_y", f"lm_{i}_z", f"lm_{i}_visibility"]
            st.session_state.csv_writer.writerow(header)
            print("CSV header written")

            # 파일 저장 정보를 확장 가능한 섹션에 표시
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

        # 메인 캡처 루프
        is_squat_down = False
        frame_count = 0
        start_time = time.time()
        KNEE_ANGLE_THRESHOLD = 120  # 무릎 구부림 감지 임계값
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
                frame = cv2.flip(frame, 1)  # 수평으로 미러링
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                status_text.write(f"### Status: Measuring")
                squat_count_text.write(f"Squat Count: **{st.session_state.current_squat_count}/5**")
                
                if frame_count % 30 == 0:  # 30프레임마다 로그
                    print(f"Processing frame: {frame_count} (Squat count: {st.session_state.current_squat_count}/5)")

                # MediaPipe로 프레임 처리
                results = st.session_state.pose.process(rgb)
                annotated_frame = frame.copy()

                if results.pose_landmarks:
                    # 랜드마크 가져오기
                    landmarks = results.pose_landmarks.landmark
                    
                    # CSV에 저장
                    row = [frame_count, current_time, st.session_state.current_squat_count]
                    for lm in landmarks:
                        row += [lm.x, lm.y, lm.z, lm.visibility]
                    st.session_state.csv_writer.writerow(row)

                    # 더 쉬운 처리를 위해 랜드마크 변환
                    frame_landmarks = [{
                        'x': lm.x,
                        'y': lm.y,
                        'z': lm.z,
                        'visibility': lm.visibility
                    } for lm in landmarks]

                    # 각도 계산 및 저장
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

                    # 포즈 랜드마크 그리기
                    mp_drawing.draw_landmarks(
                        annotated_frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                    )

                    # 각도 가져오기
                    knee_angle = frame_angles['knee']
                    hip_angle = frame_angles['hip']
                    ankle_angle = frame_angles['ankle']
                    side_used = frame_angles.get('side_used', 'right')  # 지정되지 않은 경우 오른쪽 기본값

                    # 각도 정보 표시
                    angles_text.write(f"""
                    **Measured Angles**
                    - Hip: {hip_angle:.1f}°
                    - Knee: {knee_angle:.1f}°
                    - Ankle: {ankle_angle:.1f}°
                    """)
                    
                    # 어떤 측면이 사용되고 있는지 표시
                    side_text = f"Using: {side_used.title()} Side"
                    if side_used == 'both':
                        side_text = "Using: Both Sides (Averaged)"
                    side_used_text.write(f"**{side_text}**")

                    # 프레임에 각도 텍스트 추가
                    cv2.putText(annotated_frame, f"Hip: {hip_angle:.1f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f"Knee: {knee_angle:.1f}", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f"Ankle: {ankle_angle:.1f}", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(annotated_frame, side_text, (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    # 스쿼트 감지
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
                        
                        # 스쿼트 자세 랜드마크 저장
                        st.session_state.squat_positions.append({'landmarks': frame_landmarks})
                        print(f"Squat position saved (total: {len(st.session_state.squat_positions)})")

                # 비디오에 프레임 쓰기
                st.session_state.out_raw.write(raw_frame)
                st.session_state.out_annot.write(annotated_frame)

                # 주기적으로 프레임 이미지 저장
                if frame_count % 10 == 0:  # 10프레임마다
                    img_filename_raw = os.path.join(user_folders["image"], f"frame_{timestamp}_{frame_count:04d}.jpg")
                    img_filename_annot = os.path.join(user_folders["image_anno"], f"frame_{timestamp}_{frame_count:04d}.jpg")
                    cv2.imwrite(img_filename_raw, raw_frame)
                    cv2.imwrite(img_filename_annot, annotated_frame)
                    if frame_count % 30 == 0:  # 30프레임마다 로그
                        print(f"Frame images saved: {frame_count:04d} (raw & annotated)")

                # 프레임 표시
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
                
                # 각도 이력 저장
                angles_history_path = os.path.join(user_folders["results"], f"angles_history_{timestamp}.csv")
                angles_df = pd.DataFrame([
                    {
                        'frame': item['frame'],
                        'time': item['time'],
                        'hip_angle': item['angles']['hip'],
                        'knee_angle': item['angles']['knee'],
                        'ankle_angle': item['angles']['ankle'],
                        'side_used': item['angles'].get('side_used', 'right'),  # side_used 정보 추가
                        'squat_count': item['squat_count']
                    }
                    for item in st.session_state.joint_angles_history
                ])
                angles_df.to_csv(angles_history_path, index=False)
                print(f"Angle history saved: {angles_history_path} ({len(angles_df)} rows)")
                
                with st.expander("📊 Saved Data Information", expanded=False):
                    st.write(f"**Angle History Saved**: `{angles_history_path}`")
                    st.dataframe(angles_df.head())
                
                # 평가 페이지로 이동하는 버튼
                if st.button("Go to Evaluation", key="goto_evaluation"):
                    st.session_state.active_tab = "evaluate"
                    st.rerun()
            else:
                st.info(f"Squat measurement stopped. (Current count: {st.session_state.current_squat_count})")
                print(f"Squat stopped: {st.session_state.current_squat_count} squats")

# -----------------------------------------------------------------------------------
# 13) 스쿼트 평가 함수 (OpenAI로 강화)
# -----------------------------------------------------------------------------------
def evaluate_squat():
    """AI 분석을 통한 스쿼트 자세 평가"""
    if is_cloud_env and not st.session_state.squat_positions:
        # 클라우드 환경에서 데모 목적으로 샘플 데이터 생성
        st.info("Running in cloud environment. Loading sample data for demonstration.")
        
        # 랜덤 랜드마크로 샘플 스쿼트 위치 생성
        sample_landmarks = []
        for _ in range(33):
            sample_landmarks.append({
                'x': np.random.uniform(0, 1),
                'y': np.random.uniform(0, 1),
                'z': np.random.uniform(0, 1),
                'visibility': np.random.uniform(0.7, 1.0)
            })
        
        # 합리적인 각도를 보장하기 위해 주요 랜드마크에 특정 값 사용
        # 오른쪽 측면 랜드마크
        sample_landmarks[12]['x'], sample_landmarks[12]['y'] = 0.6, 0.3  # 오른쪽 어깨
        sample_landmarks[24]['x'], sample_landmarks[24]['y'] = 0.55, 0.5  # 오른쪽 엉덩이
        sample_landmarks[26]['x'], sample_landmarks[26]['y'] = 0.57, 0.7  # 오른쪽 무릎
        sample_landmarks[28]['x'], sample_landmarks[28]['y'] = 0.52, 0.9  # 오른쪽 발목
        
        # 왼쪽 측면 랜드마크
        sample_landmarks[11]['x'], sample_landmarks[11]['y'] = 0.4, 0.3  # 왼쪽 어깨
        sample_landmarks[23]['x'], sample_landmarks[23]['y'] = 0.45, 0.5  # 왼쪽 엉덩이
        sample_landmarks[25]['x'], sample_landmarks[25]['y'] = 0.43, 0.7  # 왼쪽 무릎
        sample_landmarks[27]['x'], sample_landmarks[27]['y'] = 0.48, 0.9  # 왼쪽 발목
        
        # 약간의 변동이 있는 5개의 샘플 위치 생성
        for i in range(5):
            varied_landmarks = []
            for lm in sample_landmarks:
                # 각 랜드마크에 작은 랜덤 변동 추가
                varied_landmarks.append({
                    'x': lm['x'] + np.random.uniform(-0.02, 0.02),
                    'y': lm['y'] + np.random.uniform(-0.02, 0.02),
                    'z': lm['z'] + np.random.uniform(-0.02, 0.02),
                    'visibility': min(1.0, lm['visibility'] + np.random.uniform(-0.05, 0.05))
                })
            st.session_state.squat_positions.append({'landmarks': varied_landmarks})
            
        # 샘플 관절 각도 이력 생성
        for i in range(50):
            frame_landmarks = []
            for lm in sample_landmarks:
                # 애니메이션 효과를 위해 더 많은 변동 추가
                frame_landmarks.append({
                    'x': lm['x'] + np.random.uniform(-0.05, 0.05),
                    'y': lm['y'] + np.random.uniform(-0.05, 0.05),
                    'z': lm['z'] + np.random.uniform(-0.05, 0.05),
                    'visibility': min(1.0, lm['visibility'] + np.random.uniform(-0.1, 0.1))
                })
            
            angles = calculate_joint_angles(frame_landmarks)
            st.session_state.joint_angles_history.append({
                'frame': i,
                'time': i * 0.1,
                'angles': angles,
                'squat_count': min(4, i // 10)
            })
    
    # 활성 탭 설정
    st.session_state.active_tab = "evaluate"
    
    print("Starting squat posture evaluation with AI")
    
    # 평가 컨테이너 생성
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

        # 관절 각도 계산
        angles = calculate_joint_angles()
        st.session_state.squat_results = angles

        # 각도 분석 표시
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
        
        # 측정에 사용된 측면 표시
        if 'side_used' in angles:
            side_text = f"Using measurements from: {angles['side_used'].title()} Side"
            if angles['side_used'] == 'both':
                side_text = "Using measurements from: Both Sides (Averaged)"
            st.info(side_text)

        # 결과 저장
        result_csv_path = os.path.join(user_folders["results"], f"squat_results_{timestamp}.csv")
        df.to_csv(result_csv_path, index=False)
        print(f"Evaluation results saved: {result_csv_path}")
        with st.expander("📁 Result File Information", expanded=False):
            st.write(f"Results saved to: `{result_csv_path}`")

        # 각도 설명
        update_angle_explanation()

        # 각도 비교 시각화
        st.subheader("Angle Comparison Visualization")
        comparison_fig = generate_angle_comparison_visualization(angles, TARGET_ANGLES)
        st.pyplot(comparison_fig)
        comparison_fig_path = os.path.join(user_folders["results"], f"angle_comparison_{timestamp}.png")
        comparison_fig.savefig(comparison_fig_path, dpi=150, bbox_inches='tight')
        print(f"Angle comparison visualization saved: {comparison_fig_path}")

        # AI 분석 섹션
        st.subheader("💡 AI Posture Analysis")
        
        if st.session_state.ai_analysis is None:
            with st.spinner("AI is analyzing your squat posture... this may take a moment."):
                if OPENAI_API_KEY:
                    st.session_state.ai_analysis = get_ai_analysis(angles, TARGET_ANGLES, TOLERANCE)
                else:
                    st.session_state.ai_analysis = """
                    ## AI Analysis Not Available
                    
                    To receive AI-powered analysis of your squat form, please add your OpenAI API key in the sidebar.
                    
                    The AI analysis provides detailed feedback on your posture, specific corrections, and personalized exercise recommendations.
                    """
                
                # AI 분석 저장
                ai_analysis_path = os.path.join(user_folders["results"], f"ai_analysis_{timestamp}.txt")
                with open(ai_analysis_path, 'w', encoding='utf-8') as f:
                    f.write(st.session_state.ai_analysis)
                print(f"AI analysis saved to: {ai_analysis_path}")
        
        # 더 나은 형식으로 AI 분석 표시
        st.markdown(st.session_state.ai_analysis)

        # 기본 피드백
        st.subheader("Quick Feedback Summary")
        feedback, detailed_feedback = provide_clear_feedback(angles)
        
        if feedback:
            for i, (issue, detail) in enumerate(zip(feedback, detailed_feedback)):
                with st.expander(f"📌 {issue}", expanded=True):
                    st.info(f"{detail}")
        else:
            st.success("All joint angles are within the target range. Excellent squat posture!")

        # 각도 변화 그래프
        if st.session_state.joint_angles_history:
            print(f"Generating angle change graph (data points: {len(st.session_state.joint_angles_history)})")
            st.subheader("Joint Angle Changes During Squat Exercise")
            df_angles = pd.DataFrame([
                {
                    'time': item['time'],
                    'hip_angle': item['angles']['hip'],
                    'knee_angle': item['angles']['knee'],
                    'ankle_angle': item['angles']['ankle'],
                    'side_used': item['angles'].get('side_used', 'right'),  # side used 정보 추가
                    'squat_count': item['squat_count']
                }
                for item in st.session_state.joint_angles_history
            ])
            
            # 데이터 요약 통계
            print(f"Angle data summary:")
            print(f"  - Time range: {df_angles['time'].min():.1f}s ~ {df_angles['time'].max():.1f}s")
            print(f"  - Hip angle range: {df_angles['hip_angle'].min():.1f}° ~ {df_angles['hip_angle'].max():.1f}°")
            print(f"  - Knee angle range: {df_angles['knee_angle'].min():.1f}° ~ {df_angles['knee_angle'].max():.1f}°")
            print(f"  - Ankle angle range: {df_angles['ankle_angle'].min():.1f}° ~ {df_angles['ankle_angle'].max():.1f}°")
            
            # 측면 분포 표시
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
            
            # 스쿼트 전환 표시
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
            
            # 저장 및 표시
            angles_plot_path = os.path.join(user_folders["results"], f"angles_plot_{timestamp}.png")
            plt.savefig(angles_plot_path, dpi=150, bbox_inches='tight')
            print(f"Angle change graph saved: {angles_plot_path}")
            st.pyplot(fig)
            
            # 측면 분포를 파이 차트로 표시
            st.subheader("Measurement Side Distribution")
            fig_pie, ax_pie = plt.subplots(figsize=(6, 6))
            side_counts.plot.pie(autopct='%1.1f%%', ax=ax_pie, title='Body Side Used for Measurements')
            pie_path = os.path.join(user_folders["results"], f"side_distribution_{timestamp}.png")
            plt.savefig(pie_path, dpi=150, bbox_inches='tight')
            st.pyplot(fig_pie)

        # 스쿼트 측정 원리에 관한 정보
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

        # 관절 각도 분포 히스토그램
        if len(df_angles) > 0:
            st.subheader("Joint Angle Distribution Histograms")
            print("Generating angle distribution histograms")
            bins = 20
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # 히스토그램 생성
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
            
            # 저장 및 표시
            heatmap_path = os.path.join(user_folders["results"], f"angle_heatmap_{timestamp}.png")
            plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
            print(f"Angle distribution histograms saved: {heatmap_path}")
            st.pyplot(fig)

        # 점수 계산
        hip_score = max(0, 10 - abs(angles['hip'] - TARGET_ANGLES['hip'])) / 10 * 100
        knee_score = max(0, 10 - abs(angles['knee'] - TARGET_ANGLES['knee'])) / 10 * 100
        ankle_score = max(0, 10 - abs(angles['ankle'] - TARGET_ANGLES['ankle'])) / 10 * 100
        overall_score = (hip_score * 0.4) + (knee_score * 0.4) + (ankle_score * 0.2)
        print(f"Score calculation: Hip={hip_score:.1f}, Knee={knee_score:.1f}, Ankle={ankle_score:.1f}, Overall={overall_score:.1f}")

        # 점수 표시
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
        
        # 점수 저장
        scores_df = pd.DataFrame({
            'hip_score': [hip_score],
            'knee_score': [knee_score],
            'ankle_score': [ankle_score],
            'overall_score': [overall_score]
        })
        scores_df.to_csv(os.path.join(user_folders["results"], f"scores_{timestamp}.csv"), index=False)
        print(f"Scores saved: {os.path.join(user_folders['results'], f'scores_{timestamp}.csv')}")

        # 레이더 차트 생성
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
        
        ideal_scores = [1.0] * (len(categories)+1)
        ax.plot(angles, ideal_scores, 'o-', linewidth=2, label='Ideal Score')
        ax.fill(angles, ideal_scores, alpha=0.1)
        
        ax.set_thetagrids(np.degrees(angles[:-1]), categories[:-1])
        ax.set_ylim(0, 1)
        ax.grid(True)
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        # 저장 및 표시
        radar_chart_path = os.path.join(user_folders["results"], f"radar_chart_{timestamp}.png")
        plt.savefig(radar_chart_path, dpi=150, bbox_inches='tight')
        print(f"Radar chart saved: {radar_chart_path}")
        st.pyplot(fig)

        # 최종 정보 및 탐색 버튼
        st.write("Evaluation complete! Now you can check the 'Custom Guide' tab for personalized squat guidance with AI-generated visuals.")
        
        # 사용자 정의 가이드 페이지로 이동하는 버튼
        if st.button("Go to Custom Guide", key="goto_guide"):
            st.session_state.active_tab = "guide"
            st.rerun()

# -----------------------------------------------------------------------------------
# 14) DALL-E 이미지 생성이 포함된 사용자 정의 가이드 함수
# -----------------------------------------------------------------------------------
def generate_squat_guide():
    """AI 분석 및 DALL-E 시각화를 통한 개인화된 스쿼트 가이드 생성"""
    # 기존 데이터가 없는 클라우드 환경의 경우
    if is_cloud_env and not st.session_state.squat_results:
        evaluate_squat()  # 샘플 데이터 생성 및 평가
        
    # 활성 탭 설정
    st.session_state.active_tab = "guide"
    
    print("Starting personalized squat guide generation with AI")
    
    # 가이드 컨테이너 생성
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
        
        # 분석에 사용된 신체 측면 표시
        if 'side_used' in angles:
            side_text = f"Analysis based on: {angles['side_used'].title()} Side"
            if angles['side_used'] == 'both':
                side_text = "Analysis based on: Both Sides (Averaged)"
            st.info(side_text)

        # 가능한 경우 평가에서 AI 분석 표시
        if st.session_state.ai_analysis:
            st.subheader(f"🧠 AI Analysis for {user_name}")
            st.markdown(st.session_state.ai_analysis)
        
        # DALL-E로 개인화된 시각적 가이드 생성
        st.subheader("🖼️ Personalized Squat Visualization")
        
        if st.session_state.generated_image is None and openai_available and OPENAI_API_KEY:
            # 주요 문제 식별
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
            
            # 분석 기반 DALL-E 프롬프트 준비
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
            
            # DALL-E로 이미지 생성
            with st.spinner("Generating personalized squat guide image... this may take a moment."):
                image, image_url = generate_dalle_image(image_prompt)
                if image:
                    st.session_state.generated_image = image
                    image_path = os.path.join(user_folders["ai_images"], f"squat_guide_image_{timestamp}.png")
                    image.save(image_path)
                    print(f"Squat guide image saved: {image_path}")
                else:
                    st.error("Image generation failed. Please check your OpenAI API key.")
        
        # 생성된 이미지 표시
        if st.session_state.generated_image:
            st.image(st.session_state.generated_image, caption="AI-Generated Personalized Squat Guide", use_column_width=True)
        else:
            if not OPENAI_API_KEY:
                st.warning("⚠️ OpenAI API key not provided. Add your API key in the sidebar to generate personalized images.")
            else:
                st.warning("⚠️ Image generation failed or unavailable in this environment.")
            
            # 대신 텍스트 기반 가이드 표시
            st.subheader("📋 Text-Based Guidance")
            st.markdown("""
            Here's a text-based guide for proper squat form:
            
            1. **Foot Position**: Stand with feet shoulder-width apart, toes pointed slightly outward (15-30°)
            2. **Hip Hinge**: Begin the movement by pushing your hips back as if sitting in a chair
            3. **Knee Alignment**: Keep knees tracking over toes, not collapsing inward
            4. **Depth**: Lower until thighs are parallel to the ground (hip and knee at approximately 90°)
            5. **Back Position**: Maintain a neutral spine throughout the movement
            6. **Weight Distribution**: Keep weight centered over mid-foot, not shifting to toes
            7. **Ankle Mobility**: Allow appropriate ankle dorsiflexion (about 25°) while keeping heels on the ground
            """)
        
        # 주요 포인트
        st.subheader("💡 Key Points for Improvement")
        
        # 개선 영역 식별
        hip_diff = angles['hip'] - TARGET_ANGLES['hip']
        knee_diff = angles['knee'] - TARGET_ANGLES['knee']
        ankle_diff = angles['ankle'] - TARGET_ANGLES['ankle']
        
        # 레이아웃을 위한 열 생성
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
        
        # 보충 운동 섹션
        st.subheader("💪 Recommended Supplementary Exercises")
        
        # 식별된 문제에 따라 특정 운동 추천
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
        
        # 후속 권장 사항
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
# 15) 메인 애플리케이션 레이아웃
# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------
# 15) 메인 애플리케이션 레이아웃
# -----------------------------------------------------------------------------------
def main():
    """메인 애플리케이션 레이아웃 및 상호작용 로직"""
    # global 변수 선언을 함수 시작 부분으로 이동
    global OPENAI_API_KEY
    global client
    
    print("Loading user information...")
    st.session_state.users = load_users()
    
    # 클라우드 환경 경고 표시
    if is_cloud_env:
        st.warning("""
        ⚠️ Running in cloud environment - Camera capture features are disabled.
        
        For full functionality including real-time squat analysis with webcam, please run this application locally.
        
        Demo mode is enabled with sample data for evaluation and guide features.
        """)
    
    # 중앙 정렬 로고
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        logo_img = load_logo()
        if logo_img:
            st.image(logo_img, width=300, use_column_width=True)
        else:
            st.markdown("<div style='text-align: center;'><h2>HealthnAI Squat Analysis</h2></div>", unsafe_allow_html=True)

    st.title("AI Squat Analysis")
    st.markdown("---")

    # 사이드바에 API 키 입력
    with st.sidebar:
        st.subheader("OpenAI API Settings")
        api_key = st.text_input("Enter OpenAI API Key", value=OPENAI_API_KEY, type="password")
        if api_key != OPENAI_API_KEY:
            os.environ["OPENAI_API_KEY"] = api_key
            OPENAI_API_KEY = api_key
            # 새 API 키로 클라이언트 재초기화
            if openai_new_client:
                client = OpenAI(api_key=OPENAI_API_KEY)
            else:
                import openai
                openai.api_key = OPENAI_API_KEY
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

    # 나머지 코드는 동일하게 유지...


    # 탭 생성
    tabs = ["User Management", "Squat Measurement", "Posture Evaluation", "Custom Guide"]
    active_tab = st.session_state.active_tab if "active_tab" in st.session_state else "user"
    
    # 탭 인덱스 계산
    tab_index = 0
    if active_tab == "capture":
        tab_index = 1
    elif active_tab == "evaluate":
        tab_index = 2
    elif active_tab == "guide":
        tab_index = 3
    
    tab1, tab2, tab3, tab4 = st.tabs(tabs)
    
    # 사용자 관리 탭
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
                    # 사용자 전환 시 AI 분석 및 이미지 초기화
                    st.session_state.ai_analysis = None
                    st.session_state.generated_image = None
                    st.success(f"✅ Selected user: {st.session_state.users[selected_user_id]['name']}")
                    print(f"User selected: {st.session_state.users[selected_user_id]['name']} (ID: {selected_user_id})")
            else:
                st.info("📝 No registered users. Please register a user first.")
                print("No registered users")

        # 현재 사용자 정보 표시
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

    # 스쿼트 측정 탭
    with tab2:
        st.header("Squat Measurement")
        
        if is_cloud_env:
            st.info("💻 This feature requires a local environment with webcam access.")
            st.markdown("""
            In a local environment, this tab would allow you to:
            1. Capture real-time video from your webcam
            2. Track your squat movement using computer vision
            3. Record 5 squats and analyze each repetition
            4. Save your squat data for detailed evaluation
            
            To use this feature, please run this application locally.
            """)
            
            # 데모 목적의 샘플 데이터 버튼
            if st.session_state.current_user:
                if st.button("Generate Sample Data (Demo)", use_container_width=True):
                    # 세션 ID 생성
                    st.session_state.user_session_id = create_session_id()
                    # 카운터 초기화
                    st.session_state.current_squat_count = 5
                    st.session_state.squat_positions = []
                    st.session_state.joint_angles_history = []
                    
                    # 샘플 데이터를 생성하고 표시하기 위해 evaluate_squat 호출
                    st.session_state.active_tab = "evaluate"
                    st.rerun()
            else:
                st.warning("⚠️ Please register or select a user first.")
        else:
            if not st.session_state.current_user:
                st.warning("⚠️ Please register or select a user first.")
            else:
                st.markdown("""
                This feature measures your squat movement in real-time through the webcam. Click the Start button and perform 5 squats.
                The camera will detect your squat posture and analyze the angles of each joint in real-time.
                """)
                
                if not opencv_available:
                    st.error("OpenCV and/or MediaPipe libraries are not available. Please install them to use this feature.")
                else:
                    if st.button("📸 Start Squat Measurement", use_container_width=True):
                        print(f"Starting squat measurement (user: {st.session_state.users[st.session_state.current_user]['name']})")
                        do_capture()
                    
                # 측정 상태 및 결과 표시
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

    # 자세 평가 탭
    with tab3:
        st.header("AI Squat Posture Evaluation")
        
        if not st.session_state.current_user:
            st.warning("⚠️ Please register or select a user first.")
        elif not is_cloud_env and st.session_state.current_squat_count == 0:
            st.warning("⚠️ Please complete squat measurement first.")
        else:
            st.markdown("""
            Evaluate your squat posture with AI analysis. Your joint angles will be compared with ideal squat form angles, 
            and our AI will provide detailed professional feedback.
            """)
            
            if st.button("🔍 Start AI Posture Evaluation", use_container_width=True):
                # 새로운 분석을 보장하기 위해 AI 분석 초기화
                st.session_state.ai_analysis = None
                print(f"Starting squat evaluation")
                evaluate_squat()
                
            # 평가 결과가 있으면 가이드 버튼 표시
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

    # 사용자 정의 가이드 탭
    with tab4:
        st.header("AI-Generated Custom Squat Guide")
        
        if not st.session_state.current_user:
            st.warning("⚠️ Please register or select a user first.")
        elif not is_cloud_env and not st.session_state.squat_results:
            st.warning("⚠️ Please complete squat evaluation first.")
        else:
            st.markdown("""
            Based on your evaluation results, our AI will create a personalized squat guide, including a custom visualization
            of proper form tailored to your specific needs.
            """)
            
            if st.button("🧠 Generate AI Squat Guide & Visualization", use_container_width=True):
                print(f"Generating AI squat guide")
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

# 메인 애플리케이션 실행
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error(traceback.format_exc())