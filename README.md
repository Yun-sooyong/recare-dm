# RECARE-DM-DASHBOARD

당뇨 환자 재입원 위험 예측 대시보드

---

## 📁 프로젝트 구조

```
RECARE-DM-DASHBOARD/
├── recareDMD/          # Django config (settings, urls, wsgi)
├── dashboard/          # 박검모 — 대시보드
├── patients/           # 최일운 — 환자 목록 · 상세
├── discharge/          # 윤수용 — 퇴원 준비 · 안내 · 재방문 등록
├── prediction/         # 서정은 — 위험예측 · 대량예측
├── analysis/           # 정다희 — SHAP 분석
├── templates/          # 공통 base.html
├── static/             # CSS · JS
├── lgbm_model.pkl      # LightGBM 학습 모델
├── manage.py
└── requirements.txt
```

---

## 👥 역할 분담 & 브랜치

| 이름   | 담당                                   | 브랜치            |
| ------ | -------------------------------------- | ----------------- |
| 박검모 | 대시보드                               | `feature/geommo`  |
| 최일운 | 환자관리 1 (목록 CRUD · 상세)          | `feature/ilun`    |
| 윤수용 | 환자관리 2 (퇴원 준비 · 안내 · 재방문) | `feature/suyong`  |
| 서정은 | 위험예측 · 대량예측                    | `feature/jungeun` |
| 정다희 | SHAP 분석                              | `feature/dahee`   |

---

## 🔀 브랜치 운영 규칙

### 브랜치 구조

```
main                  ← 최종 발표용 (팀장만 merge)
└── dev               ← 통합 브랜치 (PR merge 대상)
    ├── feature/geommo
    ├── feature/ilun
    ├── feature/suyong
    ├── feature/jungeun
    └── feature/dahee
```

### 작업 흐름

```bash
# 1. 최초 1회 — clone 후 본인 브랜치로 이동
git clone https://github.com/팀저장소/RECARE-DM-DASHBOARD.git
cd RECARE-DM-DASHBOARD
git checkout feature/본인브랜치

# 2. 작업 중 — 본인 브랜치에 수시로 push (중간 저장)
git add .
git commit -m "feat: 작업 내용"
git push origin feature/본인브랜치

# 3. 모든 작업 완료 후 — GitHub에서 PR 생성
# feature/본인브랜치 → dev 로 Pull Request 올리기
# GitHub에 자동으로 뜨는 [Compare & pull request] 버튼 클릭
# 팀장이 확인 후 dev에 merge

# 4. 발표 전 — 팀장이 dev → main merge
```

### ⚠️ 충돌 방지 주의사항

**다른 사람 PR이 merge된 후 작업 재개할 때**

```bash
git pull origin dev   # dev 최신화
```

**공유 파일 수정 시 필수 공지**

- `patients/models.py` — discharge 앱이 참조하므로 변경 시 수용에게 바로 알림
- `templates/base.html` — 전체 영향, 수정 전 팀 전체 공지
- `recareDMD/settings.py` — 앱 등록 · 경로 변경 시 팀 공지
- `migrations/` 파일 — 절대 임의로 수정 금지, 충돌 시 팀장에게 문의

---

## 💬 커밋 메시지 규칙

```
feat:     새 기능 추가
fix:      버그 수정
style:    UI · 레이아웃 변경
refactor: 코드 리팩토링
docs:     문서 수정
chore:    설정 · 패키지 변경
```

예시

```
feat: 퇴원 준비 페이지 템플릿 작성
fix: 환자 목록 필터 오류 수정
style: 대시보드 차트 레이아웃 조정
```

---

## ⚙️ 로컬 실행 방법

```bash
# 1. 저장소 클론
git clone https://github.com/팀저장소/RECARE-DM-DASHBOARD.git
cd RECARE-DM-DASHBOARD

# 2. 가상환경 생성 및 활성화
python -m venv .venv
source .venv/bin/activate        # Mac/Linux
.venv\Scripts\activate           # Windows

# 3. 패키지 설치
pip install -r requirements.txt

# 4. DB 마이그레이션
python manage.py migrate

# 5. 서버 실행
python manage.py runserver
```

---

## 🚨 공통 주의사항

- `db.sqlite3` **push 금지** — DB는 개인 로컬에서 관리
- `lgbm_model.pkl` 변경 시 **팀 전체에 공유 후 교체**
- `migrations/` 파일 변경 시 반드시 팀 공유 — 충돌 위험 높음
- `base.html` 수정 시 **채팅 공유 후 진행** — 전체 영향
