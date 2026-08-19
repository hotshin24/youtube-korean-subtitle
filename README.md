# YouTube 한국어 자막 생성기

외국어 YouTube 영상이나 업로드한 영상에서 음성을 인식하고 다음 파일을 만드는 Streamlit 앱입니다.

- 타임코드가 포함된 한국어 `.srt` 자막
- 한국어 자막이 영상에 입혀진 `.mp4`

## API 키와 개인정보

번역에는 사용자가 직접 입력한 OpenAI API 키를 사용합니다.

- 배포된 웹앱에서는 API 키를 현재 브라우저 세션에서만 사용하며 서버나 GitHub에 저장하지 않습니다.
- 로컬 macOS에서는 사용자가 선택하면 로그인 키체인에 저장할 수 있습니다.
- 영상과 생성 파일은 처리 중 임시 폴더에만 존재하며 작업이 끝나면 제거됩니다.

## 로컬 실행

Python 3.12 이상과 FFmpeg가 필요합니다. macOS에서는 한국어 자막 필터가 포함된 FFmpeg를 설치하세요.

```bash
brew install python@3.12 ffmpeg@7
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud 배포

1. 이 저장소를 자신의 GitHub 계정으로 포크합니다.
2. [Streamlit Community Cloud](https://share.streamlit.io/)에서 GitHub 계정으로 로그인합니다.
3. `Create app`에서 저장소와 `app.py`를 선택합니다.
4. 별도의 OpenAI Secret을 등록하지 않고 배포합니다. 각 사용자가 앱 화면에서 자기 API 키를 입력합니다.

`packages.txt`를 통해 배포 서버에 FFmpeg와 한국어 글꼴이 설치됩니다.

## 참고

- 최초 처리 시 Whisper 모델을 내려받으므로 시간이 걸릴 수 있습니다.
- `small` 모델은 빠르고, `medium` 또는 `large-v3`는 더 정확하지만 서버 자원을 많이 사용합니다.
- MP4 출력은 영상 전체를 다시 인코딩하므로 SRT 생성보다 오래 걸립니다.
- YouTube 다운로드는 본인이 소유하거나 사용 권한이 있는 콘텐츠에만 사용하세요.
- YouTube 변경에 따라 추출이 일시적으로 실패할 수 있습니다. 이 경우 `yt-dlp` 업데이트가 필요합니다.
