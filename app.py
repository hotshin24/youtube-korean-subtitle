from __future__ import annotations

import hmac
import json
import shutil
import subprocess
import sys
import tempfile
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


def download_youtube_video(url: str, output_dir: Path) -> tuple[Path, str]:
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
    return WhisperModel(model_name, device="auto", compute_type="int8")


def transcribe(media_path: Path, model_name: str) -> tuple[list[dict], str]:
    model = load_whisper(model_name)
    raw_segments, info = model.transcribe(
        str(media_path), beam_size=5, vad_filter=True, condition_on_previous_text=True
    )
    segments = [
        {"start": item.start, "end": item.end, "source": item.text.strip()}
        for item in raw_segments
        if item.text.strip()
    ]
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

    owner_api_key = str(st.secrets.get("OWNER_OPENAI_API_KEY", "")).strip()
    owner_access_code = str(st.secrets.get("OWNER_ACCESS_CODE", "")).strip()
    key_mode = "직접 입력"
    if owner_api_key and owner_access_code:
        key_mode = st.radio(
            "API 키 선택",
            ["직접 입력", "내 저장 키 사용"],
            horizontal=True,
            help="내 저장 키는 관리자 접근 코드가 일치할 때만 사용할 수 있습니다.",
        )

    entered_api_key = ""
    owner_code = ""
    if key_mode == "내 저장 키 사용":
        owner_code = st.text_input(
            "관리자 접근 코드",
            type="password",
            help="Streamlit 서버에 저장된 본인 API 키를 불러오는 코드입니다.",
        )
        if owner_code:
            if hmac.compare_digest(owner_code, owner_access_code):
                st.success("내 저장 키를 사용합니다.")
            else:
                st.error("관리자 접근 코드가 올바르지 않습니다.")
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
    if keyring is not None and key_mode == "직접 입력":
        save_api_key = st.checkbox("macOS 키체인에 저장", value=True)
    elif key_mode == "직접 입력":
        st.info("API 키는 저장되지 않으며 이 세션의 번역 요청에만 사용됩니다.")

    owner_unlocked = (
        key_mode == "내 저장 키 사용"
        and bool(owner_code)
        and hmac.compare_digest(owner_code, owner_access_code)
    )
    api_key = owner_api_key if owner_unlocked else (entered_api_key.strip() or saved_api_key)
    whisper_model = st.selectbox("Whisper 모델", ["small", "medium", "large-v3"], index=1)
    translation_model = st.text_input("번역 모델", value="gpt-4o-mini")
    burn_in = st.checkbox("한국어 자막이 입혀진 MP4도 만들기", value=True)

source_type = st.radio("영상 입력 방식", ["유튜브 URL", "영상 파일 업로드"], horizontal=True)
url = ""
uploaded = None
if source_type == "유튜브 URL":
    url = st.text_input("유튜브 URL", placeholder="https://www.youtube.com/watch?v=...")
else:
    uploaded = st.file_uploader("영상 또는 오디오", type=["mp4", "mov", "mkv", "webm", "mp3", "m4a", "wav"])

if st.button("한국어 자막 만들기", type="primary", use_container_width=True):
    if not api_key:
        st.error("OpenAI API 키를 입력해 주세요.")
        st.stop()
    if source_type == "유튜브 URL" and not url.strip():
        st.error("유튜브 URL을 입력해 주세요.")
        st.stop()
    if source_type == "영상 파일 업로드" and uploaded is None:
        st.error("영상 또는 오디오 파일을 선택해 주세요.")
        st.stop()
    if not Path(FFMPEG_SUBTITLE_BIN).is_file() and shutil.which(FFMPEG_SUBTITLE_BIN) is None:
        st.error("자막 필터가 포함된 FFmpeg가 필요합니다.")
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
                    media_path, title = download_youtube_video(url.strip(), temp_dir)
                    original_video = media_path
                else:
                    media_path, title = save_upload(uploaded, temp_dir)
                    original_video = media_path

                status.write("음성을 인식하고 있습니다…")
                segments, language = transcribe(media_path, whisper_model)
                if not segments:
                    raise RuntimeError("영상에서 음성을 인식하지 못했습니다.")

                status.write(f"감지 언어: {language} · {len(segments)}개 자막 번역 중…")
                translated = translate_segments(segments, api_key, translation_model)
                srt_text = make_srt(translated, "ko")
                status.update(label="자막 생성 완료", state="complete", expanded=False)

            safe_title = "".join(c for c in title if c.isalnum() or c in " -_").strip()[:80] or "subtitle"
            st.success(f"{len(translated)}개 자막을 만들었습니다.")
            st.download_button(
                "한국어 SRT 다운로드",
                data=srt_text.encode("utf-8-sig"),
                file_name=f"{safe_title}.ko.srt",
                mime="application/x-subrip",
                use_container_width=True,
            )

            if burn_in and original_video is not None:
                srt_path = temp_dir / "subtitle.srt"
                output_path = temp_dir / "subtitled.mp4"
                srt_path.write_text(srt_text, encoding="utf-8")
                with st.spinner("영상에 자막을 입히는 중…"):
                    burn_subtitles(original_video, srt_path, output_path)
                st.download_button(
                    "자막 포함 MP4 다운로드",
                    data=output_path.read_bytes(),
                    file_name=f"{safe_title}.ko.mp4",
                    mime="video/mp4",
                    use_container_width=True,
                )
    except Exception as error:
        st.error(f"처리 중 오류가 발생했습니다: {error}")
