# Market Lens web

React/Vinext 화면과 Cloudflare worker/D1 스키마입니다. Python 계산 결과와 별도 뉴스 파이프라인의 입력을 받아 표시합니다. DB·뉴스·운영 인증정보는 포함하지 않습니다.

Node.js 22.13 이상에서 `npm ci`, `npm run build`, `node --test tests/*.test.mjs`로 확인합니다. `npm run dev`는 로컬 개발 서버이며 토큰이 없으면 보호 화면을 표시합니다. `.dev.vars.example`의 비밀 이름을 참고하고 실제 `.dev.vars`는 커밋하지 마세요.

`build/sites-vite-plugin.ts`, `.openai/hosting.json`, `db/`, `drizzle/`은 필수 소스입니다. hosting 설정은 로컬 예제로 실제 배포 대상이 아닙니다. npm scripts에 배포 명령은 없습니다. 설치·데이터·외부 서비스 및 운영 한계는 [프로젝트 README](../README.md)를 참고하세요.
