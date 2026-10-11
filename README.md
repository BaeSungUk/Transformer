# BERT 미니 프로젝트: 칭찬처럼 보이는 비꼼 탐지

영어 SNS 문장이 아이러니인지 분류하는 NLP 프로젝트입니다.
선정한 `google-bert/bert-base-uncased`를 그대로 Fine-tuning합니다.
단순 감정 분석에서 놓치기 쉬운 표면 표현과 문맥의 차이를 살펴봅니다.

**전체 학습과 평가를 실제로 실행했습니다.** `REPORT.md`에 제출용 한국어 보고서와
오분류 분석을 정리했습니다. 실행된 노트북에는 학습 로그와 그래프가 포함되어 있습니다.

| 모델 | 테스트 정확도 | 테스트 macro F1 |
|---|---:|---:|
| TF-IDF + 로지스틱 회귀 | 65.05% | 0.6424 |
| BERT Fine-tuning | 68.24% | 0.6801 |

일반 감사 문장에도 아이러니를 예측하는 문제가 발견되었습니다.
실패한 예측과 검증 loss 증가도 보고서에 포함했습니다.

## 실행

1. `bert_irony_project.ipynb`를 Google Colab에 업로드합니다.
2. GPU 런타임을 선택하고 위에서부터 셀을 실행합니다.
3. 입력 셀에 영어 문장을 입력하고, 빈 줄로 종료합니다.
4. 실행된 노트북과 `project_results.zip`을 다운로드합니다.

설명은 한국어, 모델 입력은 영어입니다. 인터넷과 모델 다운로드 공간이 필요합니다.
배치 크기 8에서 GPU 메모리가 부족하면 4로 줄이세요.
노트북 설치 셀 이후 import 오류가 나면 런타임을 재시작하고 설치 다음 셀부터 실행하세요.

## 과제 요건

| 요건 | 노트북 내용 |
|---|---|
| 모델·데이터 소개 | BERT Base, TweetEval irony, 공식 데이터 분할과 라벨 |
| Tokenizer 전처리 | WordPiece 예시, 길이 분포, truncation, 동적 padding |
| Fine-tuning | 분류층 추가 및 전체 가중치 3 epoch 학습 |
| 학습·성능 평가 | loss/F1 그래프, 기준 모델 비교, 혼동행렬, 오분류 |
| 직접 입력 예측 | 최적 모델 재로딩, 영어 문장 입력, 클래스별 확률 |
| 한계·개선 분석 | 맥락 부족, 도메인 차이, 과적합, 언어·길이 제한 |

## 평가 원칙

학습/검증/테스트 공식 분할을 유지합니다. 최적 체크포인트는 검증 macro F1로 선택하고,
테스트는 최종 평가에 사용합니다. 비교 모델은 TF-IDF + 로지스틱 회귀입니다.
아이러니 탐지는 감정의 긍정·부정을 판정하는 작업과 다릅니다.

`outputs/`에 실제 성능표, 학습 로그, 그래프, 테스트 예측, 오분류,
설정과 버전 정보, 학습된 모델이 생성됩니다. 노트북의 신규 예문은 설명용입니다.
실행 전에는 성능이나 예측 결과를 가정하지 않습니다.

## 출처

- [BERT 모델 카드](https://huggingface.co/google-bert/bert-base-uncased)
- [TweetEval 데이터 카드](https://huggingface.co/datasets/cardiffnlp/tweet_eval)
- [TweetEval 공식 저장소](https://github.com/cardiffnlp/tweeteval)
- [Trainer 문서 — 사용 버전](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/trainer)

## 재생성

`python tools/build_notebook.py`로 노트북을 재생성할 수 있습니다.
실행 결과가 들어 있는 노트북은 재생성 시 덮어쓰므로 별도로 보관하세요.

로컬 실행: `python tools/run_notebook.py` (설치·입력 셀 제외).
환경이 준비된 상태에서 `python tools/verify_project.py`로 저장된 모델의 추론 일치,
빈 입력·긴 입력 처리, 성능 수치와 결과 ZIP을 확인할 수 있습니다.
