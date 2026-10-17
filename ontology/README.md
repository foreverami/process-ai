# 온톨로지 입문 — 우리 프로젝트 파일로 배우기

## 1. 온톨로지가 뭔가요

**어떤 분야의 개념과 관계, 규칙을 컴퓨터가 읽을 수 있게 적어 둔 지식 모델**입니다.
데이터베이스 스키마, 용어 사전, 규칙집을 하나로 합친 것이라고 생각하면 됩니다.

우리 시스템에서는 이런 지식을 코드 대신 온톨로지에 적어 둡니다.

- "가열 스테이지 히터는 200°C까지 쓸 수 있다" (장비 지식)
- "IPA는 이소프로판올이고, 끓는점은 82.5°C다" (재료 지식)
- "목표 온도는 기판의 최대 사용 온도를 넘으면 안 된다" (규칙)
- "Torr는 압력 단위이고 1 Torr = 133.322 Pa다" (단위 지식)

코드에 숫자와 규칙을 박아 두면 바꿀 때마다 개발자가 코드를 고쳐야 합니다.
온톨로지에 두면 공정 담당자가 파일만 고쳐도 AI 프롬프트, 검증, 안전 한계에 한꺼번에 반영됩니다.

## 2. 기본 단위: 트리플

온톨로지의 모든 사실은 **주어 – 술어 – 목적어** 세 개로 이루어진 문장(트리플)입니다.

```turtle
vd:StageHeater01  vd:maxOperatingTemperature  200.0 .      # 히터01의 최대 사용 온도는 200
vd:StageHeater01  a                           vd:Heater .  # 히터01은 히터다 (a = rdf:type)
vd:Chamber01      vd:hasComponent             vd:StageHeater01 .  # 챔버01은 히터01을 가진다
```

이 문장들이 서로 이어져 그래프(지식그래프)가 됩니다. 파일 형식은 사람이 읽기 쉬운 **Turtle(.ttl)**을 씁니다.

## 3. 파일 구성

| 파일 | 종류 | 내용 |
|---|---|---|
| `vd-core.ttl` | TBox (개념) | 어떤 종류의 것이 있고 어떻게 연결되는가. 클래스와 속성 정의 |
| `vd-equipment.ttl` | ABox (사실) | 우리 장비 1호기와 부품, 한계값 |
| `vd-materials.ttl` | ABox (사실) | 용매 9종, 기판 6종, 코팅 재료 2종의 이름과 물성 |
| `vd-shapes.ttl` | SHACL 규칙 | 레시피 검증 규칙 23개 |
| `vd-data-quality.ttl` | SHACL 규칙 | 온톨로지 자체에 빠진 값이 없는지 검사 |

**TBox**는 "히터라는 것이 있고, 히터에는 최대 사용 온도가 있다"는 틀이고,
**ABox**는 "우리 장비의 히터는 200°C까지다"라는 실제 값입니다.

## 4. 핵심 개념 세 가지

**클래스(Class)**: 종류. `vd:Heater`, `vd:Solvent`, `vd:Recipe`

```turtle
vd:Heater a owl:Class ;
    rdfs:subClassOf vd:Component , sosa:Actuator ;   # 히터는 구성품이면서 액추에이터다
    rdfs:label "히터"@ko , "heater"@en .
```

**개체(Individual)**: 클래스에 속하는 실제 하나. `vd:StageHeater01`, `vd:Isopropanol`

```turtle
vd:Isopropanol a vd:Solvent ;
    skos:prefLabel "이소프로판올"@ko ;
    skos:altLabel "IPA" , "2-프로판올"@ko , "isopropyl alcohol"@en ;   # 동의어 → AI가 인식
    vd:boilingPoint 82.5 ;
    vd:flashPoint 12.0 .
```

**속성(Property)**: 관계나 값.
- 객체 속성: 개체와 개체를 잇는다 (`vd:hasComponent`, `vd:usesSolvent`)
- 데이터 속성: 개체에 값을 붙인다 (`vd:boilingPoint`, `vd:targetTemperature`)
- 주석 속성: 사람이나 소프트웨어를 위한 메모 (`rdfs:label`, `vd:jsonField`)

이 프로젝트가 정의한 주석 속성은 온톨로지와 소프트웨어를 잇는 다리입니다. 코드에는 이 값들이 없습니다.

| 주석 속성 | 뜻 | 누가 읽나 |
|---|---|---|
| `vd:jsonField` | 레시피 JSON에서의 경로 | 추출 항목 목록, 레시피 변환 |
| `vd:required` | 작업자가 꼭 말해야 하는 값 (AI가 추측하지 않음) | 되묻기 |
| `vd:extractionHint` | LLM에게 주는 항목별 추출 지침. 작은 모델은 부정문("X는 아니다")을 보면 오히려 X를 넣으므로 긍정문으로 쓴다 | 추출 프롬프트 |
| `vd:valuePattern` | 문자열 값의 형식 정규식 (예: 시편 ID는 숫자가 든 코드) | 근거 검사 |
| `vd:code` | 소프트웨어가 개체를 가리키는 문자열 (`heat_first`) | 공정 순서 |

`vd:Coating`(PEDOT:PSS, 포토레지스트 같은 박막 재료)과 `vd:GenericTerm`("시편", "기판", "없음" 같은 일반 명사)도 같은 목적입니다. LLM이 이런 말을 재료 이름 칸에
넣으면 근거 검사가 뺍니다. 모델 평가에서 실제로 나온 실수를 보고 하나씩 추가했습니다.

## 5. 표준 어휘를 재사용하는 이유

직접 다 만들지 않고 국제 표준 어휘를 가져다 씁니다. 다른 시스템과 데이터를 주고받을 수 있고,
기업이 온톨로지를 도입할 때 가장 먼저 보는 부분이기도 합니다.

| 어휘 | 만든 곳 | 우리가 쓰는 곳 |
|---|---|---|
| SOSA/SSN | W3C | 센서(`sosa:Sensor`), 액추에이터, 관측 대상 |
| PROV-O | W3C | 공정 실행 이력: 누가(`prov:wasAssociatedWith`) 어떤 레시피로(`prov:used`) 언제 |
| QUDT | NASA 등 | 단위와 변환 배수 (`unit:TORR`, `qudt:conversionMultiplier`) |
| SKOS | W3C | 대표 이름과 동의어 (`skos:prefLabel`, `skos:altLabel`) |
| ISA-88 | 국제 배치 공정 표준 | 레시피 개념 (`vd:Recipe` = 제어 레시피) |

## 6. SHACL: 규칙도 데이터다

SHACL은 "그래프가 이런 모양이어야 한다"를 적는 W3C 표준 규칙 언어입니다.
`vd-shapes.ttl`의 규칙 하나를 보면:

```turtle
vds:TemperatureOverSubstrate a sh:PropertyShape ;
    sh:path vd:targetTemperature ;
    sh:sparql [
        sh:message "목표 온도 {?value}°C가 {?name}의 최대 사용 온도 {?limit}°C를 넘습니다."@ko ;
        sh:select """
            SELECT $this ?value ?limit ?name WHERE {
                $this $PATH ?value ; vd:usesSubstrate ?s .          # 레시피의 목표 온도와 기판
                ?s vd:maxOperatingTemperature ?limit ; skos:prefLabel ?name .   # 기판의 최대 온도
                FILTER (?value > xsd:double(?limit))                 # 넘으면 위반
            }""" ] .
```

규칙 안에 숫자가 없다는 점이 중요합니다. PET의 120°C는 `vd-materials.ttl`에 있고,
새 기판을 추가하면 이 규칙이 그대로 적용됩니다.

심각도는 세 가지입니다. `sh:Violation`(실행 불가), `sh:Warning`(주의), `sh:Info`(참고)

## 7. SPARQL: 그래프에 묻는 질의어

SQL이 표에 묻는다면 SPARQL은 그래프에 묻습니다. 장비 최대 온도를 정하는 부품을 찾는 질의:

```sparql
SELECT ?limit ?part WHERE {
    vd:Chamber01 vd:hasComponent ?c .
    ?c vd:maxOperatingTemperature ?limit ; rdfs:label ?part .
} ORDER BY ?limit LIMIT 1
```

파이썬에서 직접 해볼 수 있습니다.

```python
from process_ai.knowledge import KnowledgeBase
kb = KnowledgeBase()
for row in kb.graph.query("SELECT ?name ?bp WHERE { ?s a vd:Solvent ; skos:prefLabel ?name ; vd:boilingPoint ?bp . FILTER(lang(?name)='ko') }"):
    print(row.name, row.bp)
```

## 8. 이 시스템에서 온톨로지가 하는 일

```
작업자 문장 ─► ① 추출 항목·프롬프트 생성 (레시피 파라미터의 이름·동의어·단위)
             ─► 로컬 LLM이 값과 단위를 적힌 그대로 뽑음
             ─► ② 용어 연결  "IPA" → vd:Isopropanol,  "최대 진공" → 펌프 최저 압력
             ─► ③ 단위 변환  10 Torr → 1.333 kPa (QUDT 변환 배수)
             ─► ④ 규칙 검증  SHACL 23개 (장비·재료 한계)
             ─► ⑤ 지식 안내  끓는점, 인화점, 압력별 끓는점(Antoine 식), 과거 유사 실험
             ─► 작업자 승인 ─► 공정 제어기 (안전 한계도 온톨로지에서 읽음)
             ─► ⑥ 실행 이력을 PROV-O 지식그래프로 누적
```

작은 로컬 LLM에게는 ①의 추출만 맡기고, ②~⑥의 의미 해석과 판단은 온톨로지가 합니다.
그래서 3~4B 모델로도 안정적으로 동작하고, 결과를 설명할 수 있습니다.

## 9. 직접 해보기

**온톨로지 검사** — 파일을 고친 뒤에는 항상 실행하세요. 서버도 시작할 때 이 검사를 통과해야 뜹니다.

```bash
python3 -m process_ai.knowledge check
python3 -m process_ai.knowledge limits
```

**새 용매 추가** — `vd-materials.ttl`에 아래 형식으로 추가하고 `check`를 실행합니다.
물성값은 사용하는 제품의 SDS에서 확인해서 넣으세요.

```turtle
vd:NewSolvent a vd:Solvent ;
    skos:prefLabel "용매 한국어 이름"@ko ;
    skos:altLabel "약어" , "영문 이름"@en ;
    vd:casNumber "000-00-0" ;
    vd:boilingPoint 0.0 ;
    vd:flashPoint 0.0 .
```

서버를 다시 시작하면 채팅이 약어를 알아듣고, 양식의 용매 목록과 인화점 안내에도 바로 나타납니다.

**Protégé로 보기** — [Protégé](https://protege.stanford.edu)는 스탠퍼드에서 만든 무료 온톨로지 편집기입니다.

```bash
python3 -m process_ai.knowledge export knowledge.ttl
```

만든 `knowledge.ttl`을 Protégé에서 열면 클래스 계층(Entities 탭)과 관계 그래프(OntoGraf 탭)를 볼 수 있습니다.
콘솔 상단의 "지식그래프 내보내기"로 받은 파일도 같습니다. 여기에는 실행 이력까지 들어 있습니다.
Protégé는 SHACL 검사를 기본으로 하지 않으므로 규칙 검사는 위의 `check` 명령을 쓰세요.

## 10. 더 나아가기

- **그래프 데이터베이스**: 같은 Turtle 파일을 GraphDB나 Stardog(RDF 전용 DB)에 넣거나,
  Neo4j의 neosemantics(n10s) 플러그인으로 불러올 수 있습니다. 데이터가 많아지면 고려하세요.
- **자연어 질의**: "지난주에 비상 정지된 공정 보여줘"를 LLM이 SPARQL로 바꿔 지식그래프에 묻는 기능.
  기업에서 많이 시도하는 GraphRAG 방식이며, 이 프로젝트의 다음 단계로 좋습니다.
