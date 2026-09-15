from __future__ import annotations

import hmac
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import streamlit as st
import yt_dlp
from faster_whisper import WhisperModel
from openai import OpenAI

if sys.platform == "darwin":
    import keyring
else:
    keyring = None


st.set_page_config(page_title="유튜브 한국어 자막", page_icon="🎬", layout="centered")
st.markdown(
    """
    <style>
    [data-testid="stAppDeployButton"],
    [data-testid="stToolbarActions"],
    #MainMenu { display: none !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

KEYCHAIN_SERVICE = "youtube-korean-subtitle"
KEYCHAIN_ACCOUNT = "openai-api-key"
LOCAL_FFMPEG = Path("/opt/homebrew/opt/ffmpeg@7/bin/ffmpeg")
FFMPEG_SUBTITLE_BIN = str(LOCAL_FFMPEG) if LOCAL_FFMPEG.is_file() else (shutil.which("ffmpeg") or "ffmpeg")


def format_srt_time(seconds: float) -> str:
    millis = max(0, round(seconds * 1000))
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def make_srt(segments: list[dict], field: str) -> str:
    blocks = []
    for index, segment in enumerate(segments, 1):
        text = segment[field].strip()
        blocks.append(
            f"{index}\n{format_srt_time(segment['start'])} --> "
            f"{format_srt_time(segment['end'])}\n{text}"
        )
    return "\n\n".join(blocks) + "\n"


def download_youtube_video(
    url: str, output_dir: Path, cookie_browser: str | None = None
) -> tuple[Path, str]:
    options = {
        # mweb에서 실제 다운로드 가능한 호환 MP4(일반적으로 360p)를 선택한다.
        # 자막 생성이 목적이므로 고해상도보다 안정성과 처리 속도를 우선한다.
        "format": "best[ext=mp4]/best",
        "outtmpl": str(output_dir / "source.%(ext)s"),
        "noplaylist": True,
        "js_runtimes": {"node": {}},
        # mweb은 현재 토큰 없는 고음질 스트림을 건너뛰고 실제로 재생 가능한
        # 호환 포맷으로 자동 폴백한다. 기본 android_vr 오디오는 일부 IP에서
        # 목록에는 보이지만 다운로드 시 403을 반환할 수 있다.
        "extractor_args": {"youtube": {"player_client": ["mweb"]}},
        "quiet": True,
    }
    # 로컬 Mac에서만 로그인된 브라우저 쿠키를 직접 읽는다.
    # 쿠키 파일을 만들거나 외부 서버로 전송하지 않는다.
    if cookie_browser:
        options["cookiesfrombrowser"] = (cookie_browser,)
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=True)
    candidates = [path for path in output_dir.glob("source.*") if path.suffix not in {".part", ".ytdl"}]
    if not candidates:
        raise RuntimeError("다운로드한 영상 파일을 찾지 못했습니다.")
    media_path = next((path for path in candidates if path.suffix.lower() == ".mp4"), candidates[0])
    return media_path, info.get("title", "youtube-video")


def save_upload(uploaded_file, output_dir: Path) -> tuple[Path, str]:
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    path = output_dir / f"upload{suffix}"
    path.write_bytes(uploaded_file.getbuffer())
    return path, Path(uploaded_file.name).stem


@st.cache_resource(show_spinner=False)
def load_whisper(model_name: str) -> WhisperModel:
    return WhisperModel(
        model_name,
        device="cpu",
        compute_type="int8",
        cpu_threads=4,
        num_workers=1,
    )


def transcribe(
    media_path: Path,
    model_name: str,
    beam_size: int,
    progress_callback=None,
) -> tuple[list[dict], str]:
    model = load_whisper(model_name)
    raw_segments, info = model.transcribe(
        str(media_path),
        beam_size=beam_size,
        vad_filter=True,
        condition_on_previous_text=True,
        word_timestamps=True,
    )
    segments = []
    total_duration = max(float(info.duration or 0), 1.0)
    sentence_endings = (".", "!", "?", "。", "！", "？")
    for item in raw_segments:
        words = [
            word
            for word in (item.words or [])
            if word.word.strip() and word.start is not None and word.end is not None
        ]
        if not words:
            source = item.text.strip()
            if source:
                segments.append(
                    {"start": float(item.start), "end": float(item.end), "source": source}
                )
        else:
            # Whisper의 긴 문장 단위 대신 단어 시각·침묵·문장 부호를 기준으로
            # 화면에 읽기 좋은 짧은 자막 단위로 다시 묶는다.
            cue_words = []
            for index, word in enumerate(words):
                cue_words.append(word)
                cue_start = float(cue_words[0].start)
                cue_end = float(word.end)
                next_pause = 0.0
                if index + 1 < len(words):
                    next_pause = max(0.0, float(words[index + 1].start) - cue_end)

                text = "".join(part.word for part in cue_words).strip()
                duration = cue_end - cue_start
                should_split = (
                    next_pause >= 0.55
                    or duration >= 4.5
                    or len(cue_words) >= 14
                    or (
                        text.endswith(sentence_endings)
                        and duration >= 1.2
                        and len(cue_words) >= 4
                    )
                )
                if should_split or index == len(words) - 1:
                    # 말이 끝나기 전에 자막이 사라져 보이지 않도록 아주 짧은 여유만 준다.
                    padded_end = min(cue_end + 0.12, float(words[index + 1].start)) if index + 1 < len(words) else cue_end + 0.12
                    segments.append(
                        {
                            "start": max(0.0, cue_start - 0.04),
                            "end": max(cue_start + 0.35, padded_end),
                            "source": text,
                        }
                    )
                    cue_words = []
        if progress_callback is not None:
            progress_callback(min(float(item.end) / total_duration, 1.0))
    return segments, info.language


def translate_batch(client: OpenAI, texts: list[str], model: str) -> list[str]:
    pending = {i: text for i, text in enumerate(texts)}
    translated: dict[int, str] = {}

    # 모델이 긴 JSON 응답에서 항목을 빠뜨리는 경우가 있어 누락분만 최대 3회 재요청한다.
    for _ in range(3):
        if not pending:
            break
        numbered = [{"id": item_id, "text": text} for item_id, text in pending.items()]
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "영상 자막 번역가입니다. 입력 문장을 자연스럽고 간결한 한국어 자막으로 "
                        "번역하세요. 모든 id를 정확히 한 번씩 포함하고 설명은 추가하지 마세요. "
                        "반드시 {\"translations\":[{\"id\":0,\"text\":\"...\"}]} 형태의 JSON만 반환하세요."
                    ),
                },
                {"role": "user", "content": json.dumps(numbered, ensure_ascii=False)},
            ],
        )
        try:
            data = json.loads(response.choices[0].message.content or "{}")
            for item in data.get("translations", []):
                item_id = int(item["id"])
                text = str(item["text"]).strip()
                if item_id in pending and text:
                    translated[item_id] = text
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass
        pending = {item_id: text for item_id, text in pending.items() if item_id not in translated}

    # 드물게 계속 누락되는 문장은 JSON 묶음 대신 한 문장씩 번역한다.
    for item_id, text in pending.items():
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": "다음 영상 자막을 자연스럽고 간결한 한국어로만 번역하세요."},
                {"role": "user", "content": text},
            ],
        )
        translated[item_id] = (response.choices[0].message.content or "").strip()

    return [translated[i] for i in range(len(texts))]


def translate_segments(segments: list[dict], api_key: str, model: str) -> list[dict]:
    client = OpenAI(api_key=api_key)
    result = [dict(item) for item in segments]
    batch_size = 25
    for start in range(0, len(result), batch_size):
        batch = result[start : start + batch_size]
        translations = translate_batch(client, [item["source"] for item in batch], model)
        for item, translated in zip(batch, translations):
            item["ko"] = translated
    return result


def burn_subtitles(source: Path, srt_path: Path, output_path: Path) -> None:
    escaped = str(srt_path).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
    command = [
        FFMPEG_SUBTITLE_BIN,
        "-y",
        "-i",
        str(source),
        "-vf",
        f"subtitles='{escaped}':force_style='FontName=Apple SD Gothic Neo,FontSize=20,Outline=2'",
        "-c:a",
        "copy",
        str(output_path),
    ]
    process = subprocess.run(command, capture_output=True, text=True)
    if process.returncode != 0:
        details = process.stderr.strip().splitlines()
        message = "\n".join(details[-8:]) if details else f"FFmpeg 오류 코드 {process.returncode}"
        raise RuntimeError(f"영상에 자막을 입히지 못했습니다:\n{message}")


st.title("🎬 외국어 영상 → 한국어 자막")
st.caption("음성은 Whisper로 인식하고, 번역은 사용자가 입력한 OpenAI API 키로 처리합니다.")

page = st.radio(
    "페이지",
    ["자막 만들기", "FAQ"],
    horizontal=True,
    label_visibility="collapsed",
)

if page == "FAQ":
    st.header("자주 묻는 질문")
    faq_items = [
        (
            "영어 외에 다른 나라 언어도 가능한가요?",
            "네. Whisper가 영상의 언어를 자동으로 감지하므로 일본어, 중국어, 스페인어, "
            "프랑스어, 독일어, 러시아어, 아랍어, 베트남어 등 다양한 언어를 한국어로 번역할 수 있습니다.",
        ),
        (
            "OpenAI API 키는 왜 필요한가요?",
            "인식된 외국어 자막을 자연스러운 한국어로 번역할 때 사용합니다. "
            "다른 사용자는 각자 자신의 키를 입력하며, 입력한 키는 현재 브라우저 세션에만 사용됩니다.",
        ),
        (
            "영상 처리가 오래 걸리는 이유는 무엇인가요?",
            "영상 다운로드, 음성 인식, 한국어 번역, MP4 자막 입히기를 순서대로 처리하기 때문입니다. "
            "영상 길이와 Whisper 모델 크기에 따라 시간이 달라집니다.",
        ),
        (
            "어떤 Whisper 모델을 선택하면 좋나요?",
            "빠른 처리는 small, 속도와 정확도의 균형은 medium, 정확도를 가장 중시하면 large-v3를 권장합니다. "
            "large-v3는 처리 시간이 더 오래 걸립니다.",
        ),
        (
            "SRT 파일은 어떻게 사용하나요?",
            "YouTube 자막 관리 화면이나 VLC 같은 동영상 플레이어에서 자막 파일로 불러올 수 있습니다. "
            "영상과 SRT의 파일명을 같게 두면 많은 플레이어가 자동으로 인식합니다.",
        ),
        (
            "자막이 포함된 MP4도 받을 수 있나요?",
            "네. ‘한국어 자막이 입혀진 MP4도 만들기’를 선택하면 자막이 영상에 직접 표시된 MP4를 다운로드할 수 있습니다.",
        ),
        (
            "유튜브 영상 다운로드 오류가 발생하면 어떻게 하나요?",
            "일부 연령 제한, 로그인 필요, 지역 제한, 비공개 영상은 다운로드되지 않을 수 있습니다. "
            "이 경우 직접 보유한 영상 파일을 업로드해 처리해 주세요.",
        ),
        (
            "자막 정확도를 높이려면 어떻게 해야 하나요?",
            "음성이 선명하고 배경음이 적은 영상을 사용하고 large-v3 모델을 선택하세요. "
            "강한 억양, 겹치는 대화, 전문용어가 많은 영상은 결과를 직접 검토하는 것이 좋습니다.",
        ),
    ]
    for question, answer in faq_items:
        with st.expander(question):
            st.write(answer)
    st.stop()

with st.sidebar:
    st.header("설정")
    saved_api_key = ""
    if keyring is not None:
        try:
            saved_api_key = keyring.get_password(KEYCHAIN_SERVICE, KEYCHAIN_ACCOUNT) or ""
        except Exception:
            saved_api_key = ""

    if saved_api_key:
        st.success("API 키가 macOS 키체인에 저장되어 있습니다.")
        if st.button("키체인에서 API 키 삭제", use_container_width=True):
            try:
                keyring.delete_password(KEYCHAIN_SERVICE, KEYCHAIN_ACCOUNT)
                st.rerun()
            except Exception as error:
                st.error(f"키를 삭제하지 못했습니다: {error}")

    try:
        owner_api_key = str(st.secrets.get("OWNER_OPENAI_API_KEY", "")).strip()
        owner_access_code = str(st.secrets.get("OWNER_ACCESS_CODE", "")).strip()
    except Exception:
        # 로컬 앱에는 .streamlit/secrets.toml이 없어도 정상 실행되어야 한다.
        owner_api_key = ""
        owner_access_code = ""
    admin_requested = st.query_params.get("admin") == "1"
    st.session_state.setdefault("owner_authenticated", False)
    st.session_state.setdefault("owner_failed_attempts", 0)
    st.session_state.setdefault("owner_locked_until", 0.0)

    if time.time() >= st.session_state.owner_locked_until:
        st.session_state.owner_locked_until = 0.0
        if st.session_state.owner_failed_attempts >= 5:
            st.session_state.owner_failed_attempts = 0

    if admin_requested and owner_api_key and owner_access_code and not st.session_state.owner_authenticated:
        locked_seconds = max(0, int(st.session_state.owner_locked_until - time.time()))
        if locked_seconds > 0:
            st.error(f"로그인 시도가 잠겼습니다. {locked_seconds // 60 + 1}분 후 다시 시도해 주세요.")
        else:
            with st.form("owner_login_form"):
                owner_code = st.text_input("관리자 접근 코드", type="password")
                login_submitted = st.form_submit_button("관리자 로그인", use_container_width=True)
            if login_submitted:
                if hmac.compare_digest(owner_code, owner_access_code):
                    st.session_state.owner_authenticated = True
                    st.session_state.owner_failed_attempts = 0
                    st.rerun()
                else:
                    st.session_state.owner_failed_attempts += 1
                    remaining = 5 - st.session_state.owner_failed_attempts
                    if remaining <= 0:
                        st.session_state.owner_locked_until = time.time() + 900
                        st.error("로그인 시도가 15분 동안 잠겼습니다.")
                    else:
                        st.error(f"접근 코드가 올바르지 않습니다. 남은 시도: {remaining}회")

        # 인증 전에는 일반 설정과 작업 화면을 함께 노출하지 않는다.
        st.stop()

    owner_unlocked = bool(
        admin_requested
        and st.session_state.owner_authenticated
        and owner_api_key
        and owner_access_code
    )

    if owner_unlocked:
        st.success("관리자 모드 · 저장된 API 키 사용 중")
        if st.button("관리자 모드 종료", use_container_width=True):
            st.session_state.owner_authenticated = False
            st.query_params.clear()
            st.rerun()
        entered_api_key = ""
        save_api_key = False
        api_key = owner_api_key
    else:
        entered_api_key = st.text_input(
            "새 OpenAI API 키" if saved_api_key else "OpenAI API 키",
            value="",
            type="password",
            placeholder="저장된 키를 교체할 때만 입력" if saved_api_key else "sk-...",
            help=(
                "로컬 macOS에서는 키체인에 저장할 수 있습니다."
                if keyring is not None
                else "키는 현재 브라우저 세션에서만 사용되며 서버에 저장되지 않습니다."
            ),
        )
        save_api_key = False
        if keyring is not None:
            save_api_key = st.checkbox("macOS 키체인에 저장", value=True)
        else:
            st.info("API 키는 저장되지 않으며 이 세션의 번역 요청에만 사용됩니다.")
        api_key = entered_api_key.strip() or saved_api_key

    quality_profiles = {
        "빠른 모드": {
            "model": "small",
            "beam_size": 1,
            "help": "가장 빠릅니다. 짧고 선명한 음성에 적합합니다.",
        },
        "권장 모드": {
            "model": "medium",
            "beam_size": 2,
            "help": "속도와 음성 인식 정확도의 균형이 좋습니다.",
        },
        "최고 품질": {
            "model": "large-v3",
            "beam_size": 2,
            "help": "억양·잡음·전문용어에 더 강하지만 가장 오래 걸립니다.",
        },
    }
    quality_mode = st.selectbox("처리 품질", list(quality_profiles), index=1)
    quality_profile = quality_profiles[quality_mode]
    st.caption(quality_profile["help"])
    whisper_model = quality_profile["model"]
    whisper_beam_size = quality_profile["beam_size"]
    translation_model = st.text_input(
        "번역 모델",
        value="gpt-4.1",
        help="품질 우선 기본값입니다. 비용을 줄이려면 gpt-4o-mini로 바꿀 수 있습니다.",
    )
    burn_in = st.checkbox("한국어 자막이 입혀진 MP4도 만들기", value=True)

    cookie_browser = None
    cookie_browser_label = "로그인 쿠키 사용 안 함"
    if sys.platform == "darwin":
        browser_labels = {
            "Chrome (추천)": "chrome",
            "Safari": "safari",
            "Firefox": "firefox",
            "Brave": "brave",
            "Microsoft Edge": "edge",
            "로그인 쿠키 사용 안 함": None,
        }
        cookie_browser_label = st.selectbox(
            "YouTube 로그인 브라우저",
            list(browser_labels),
            help=(
                "선택한 브라우저에서 YouTube에 로그인해 두세요. "
                "쿠키는 이 Mac 안에서만 읽으며 서버에 업로드하지 않습니다."
            ),
        )
        cookie_browser = browser_labels[cookie_browser_label]

source_type = st.radio("영상 입력 방식", ["유튜브 URL", "영상 파일 업로드"], horizontal=True)
url = ""
uploaded = None
if source_type == "유튜브 URL":
    url = st.text_input("유튜브 URL", placeholder="https://www.youtube.com/watch?v=...")
else:
    uploaded = st.file_uploader("영상 또는 오디오", type=["mp4", "mov", "mkv", "webm", "mp3", "m4a", "wav"])

st.session_state.setdefault("transcription_job", None)
st.session_state.setdefault("translation_result", None)

if st.button("1단계: 음성 인식 시작", type="primary", use_container_width=True):
    if source_type == "유튜브 URL" and not url.strip():
        st.error("유튜브 URL을 입력해 주세요.")
        st.stop()
    if source_type == "영상 파일 업로드" and uploaded is None:
        st.error("영상 또는 오디오 파일을 선택해 주세요.")
        st.stop()

    if keyring is not None and entered_api_key.strip() and save_api_key:
        try:
            keyring.set_password(KEYCHAIN_SERVICE, KEYCHAIN_ACCOUNT, entered_api_key.strip())
        except Exception as error:
            st.error(f"API 키를 macOS 키체인에 저장하지 못했습니다: {error}")
            st.stop()

    try:
        with tempfile.TemporaryDirectory(prefix="ko-subtitle-") as temp_name:
            temp_dir = Path(temp_name)
            with st.status("영상 준비 중…", expanded=True) as status:
                if source_type == "유튜브 URL":
                    if cookie_browser:
                        status.write(
                            f"{cookie_browser_label}의 YouTube 로그인 정보를 확인하고 있습니다…"
                        )
                    media_path, title = download_youtube_video(
                        url.strip(), temp_dir, cookie_browser
                    )
                else:
                    media_path, title = save_upload(uploaded, temp_dir)

                status.write("음성을 인식하고 있습니다. 영상 길이에 따라 몇 분 걸릴 수 있습니다…")
                transcription_progress = st.progress(0, text="음성 인식 0%")
                segments, language = transcribe(
                    media_path,
                    whisper_model,
                    whisper_beam_size,
                    lambda ratio: transcription_progress.progress(
                        int(ratio * 100),
                        text=f"음성 인식 {int(ratio * 100)}%",
                    ),
                )
                transcription_progress.empty()
                if not segments:
                    raise RuntimeError("영상에서 음성을 인식하지 못했습니다.")

                st.session_state.transcription_job = {
                    "media": media_path.read_bytes(),
                    "suffix": media_path.suffix or ".mp4",
                    "title": title,
                    "segments": segments,
                    "language": language,
                    "burn_in": burn_in,
                    "translation_model": translation_model,
                    "quality_mode": quality_mode,
                }
                st.session_state.translation_result = None
                status.update(label="음성 인식 완료", state="complete", expanded=False)
        st.rerun()
    except Exception as error:
        st.error(f"음성 인식 중 오류가 발생했습니다: {error}")

job = st.session_state.transcription_job
if job:
    st.divider()
    st.subheader("2단계: 인식 원문 확인")
    st.caption(
        f"감지 언어: {job['language']} · {job['quality_mode']} · "
        f"{len(job['segments'])}개 자막 — 잘못 들은 부분을 표에서 직접 고쳐 주세요."
    )
    editor_rows = [
        {
            "시작(초)": round(item["start"], 2),
            "끝(초)": round(item["end"], 2),
            "인식 원문": item["source"],
        }
        for item in job["segments"]
    ]
    edited_rows = st.data_editor(
        editor_rows,
        hide_index=True,
        use_container_width=True,
        column_config={
            "시작(초)": st.column_config.NumberColumn(
                format="%.2f", min_value=0.0, step=0.1,
                help="자막이 늦으면 값을 줄이고, 빠르면 값을 늘리세요.",
            ),
            "끝(초)": st.column_config.NumberColumn(
                format="%.2f", min_value=0.0, step=0.1,
            ),
            "인식 원문": st.column_config.TextColumn(width="large"),
        },
        key="transcript_editor",
    )

    translate_col, reset_col = st.columns([3, 1])
    with translate_col:
        translate_clicked = st.button(
            "확인한 원문으로 한국어 번역",
            type="primary",
            use_container_width=True,
        )
    with reset_col:
        if st.button("처음부터", use_container_width=True):
            st.session_state.transcription_job = None
            st.session_state.translation_result = None
            st.rerun()

    if translate_clicked:
        if not api_key:
            st.error("OpenAI API 키를 입력해 주세요.")
            st.stop()
        if job["burn_in"] and not Path(FFMPEG_SUBTITLE_BIN).is_file() and shutil.which(FFMPEG_SUBTITLE_BIN) is None:
            st.error("자막 필터가 포함된 FFmpeg가 필요합니다.")
            st.stop()

        try:
            corrected_segments = [
                {
                    "start": float(edited["시작(초)"]),
                    "end": float(edited["끝(초)"]),
                    "source": str(edited["인식 원문"]).strip(),
                }
                for edited in edited_rows
                if str(edited["인식 원문"]).strip()
            ]
            with st.status("한국어로 번역 중…", expanded=True) as status:
                status.write(
                    f"{len(corrected_segments)}개 자막을 {job['translation_model']} 모델로 번역하고 있습니다…"
                )
                for item in corrected_segments:
                    if item["end"] <= item["start"]:
                        raise ValueError("자막 종료 시간은 시작 시간보다 늦어야 합니다.")

                translated = translate_segments(
                    corrected_segments, api_key, job["translation_model"]
                )
                srt_text = make_srt(translated, "ko")
                safe_title = "".join(
                    c for c in job["title"] if c.isalnum() or c in " -_"
                ).strip()[:80] or "subtitle"

                mp4_bytes = None
                if job["burn_in"]:
                    status.write("영상에 한국어 자막을 입히고 있습니다…")
                    with tempfile.TemporaryDirectory(prefix="ko-subtitle-render-") as temp_name:
                        temp_dir = Path(temp_name)
                        source_path = temp_dir / f"source{job['suffix']}"
                        srt_path = temp_dir / "subtitle.srt"
                        output_path = temp_dir / "subtitled.mp4"
                        source_path.write_bytes(job["media"])
                        srt_path.write_text(srt_text, encoding="utf-8")
                        burn_subtitles(source_path, srt_path, output_path)
                        mp4_bytes = output_path.read_bytes()

                st.session_state.translation_result = {
                    "count": len(translated),
                    "srt": srt_text.encode("utf-8-sig"),
                    "mp4": mp4_bytes,
                    "safe_title": safe_title,
                }
                status.update(label="자막 생성 완료", state="complete", expanded=False)
        except Exception as error:
            st.error(f"번역 또는 영상 생성 중 오류가 발생했습니다: {error}")

result = st.session_state.translation_result
if result:
    st.success(f"{result['count']}개 자막을 만들었습니다.")
    st.download_button(
        "한국어 SRT 다운로드",
        data=result["srt"],
        file_name=f"{result['safe_title']}.ko.srt",
        mime="application/x-subrip",
        use_container_width=True,
    )
    if result["mp4"] is not None:
        st.download_button(
            "자막 포함 MP4 다운로드",
            data=result["mp4"],
            file_name=f"{result['safe_title']}.ko.mp4",
            mime="video/mp4",
            use_container_width=True,
        )
