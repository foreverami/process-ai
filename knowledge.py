"""온톨로지 기반 지식 계층.

ontology/의 RDF(Turtle) 파일을 읽어 다음 일을 한다.
- 장비 한계값 조회 → 공정 제어기의 안전 인터락          limits()
- 용어 연결: "IPA", "이소프로판올" → vd:Isopropanol     link_entities(), link_name()
- 단위 변환: 5 Torr → 0.667 kPa                         to_canonical()
- 레시피 검증: SHACL 규칙                               validate()
- 물성 안내, 과거 유사 실험 조회: SPARQL                facts(), similar_runs()
- 실행 기록을 PROV-O 지식그래프로 변환                   runs_graph()

명령줄:  python3 -m process_ai.knowledge check | limits | export [파일]
"""
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from pydantic import BaseModel
from pyshacl import validate as shacl_validate
from rdflib import RDF, RDFS, XSD, Graph, Literal, Namespace, URIRef
from rdflib.namespace import PROV, SKOS

from .config import RUNS_DIR, EquipmentLimits
from .recipe import Issue, Recipe, field_paths

VD = Namespace("http://example.org/vacuum-drying#")
SH = Namespace("http://www.w3.org/ns/shacl#")
QUDT = Namespace("http://qudt.org/schema/qudt/")

ONTOLOGY_DIR = Path(__file__).parent / "ontology"
KNOWLEDGE_FILES = ("vd-core.ttl", "vd-equipment.ttl", "vd-materials.ttl")
MMHG_PER_KPA = 7.50062
SIMILAR_TEMP_RANGE_C = 15.0

SEVERITY = {SH.Violation: "error", SH.Warning: "warning", SH.Info: "info"}
LEVEL_ORDER = {"error": 0, "warning": 1, "info": 2}
LINKABLE_TYPES = (VD.Solvent, VD.Substrate, VD.Coating, VD.QualitativeSetpoint, VD.ProcessOrder, VD.GenericTerm)
MATERIAL_TYPES = (VD.Solvent, VD.Substrate, VD.Coating)
KOREAN_PARTICLES = "을|를|이|가|은|는|의|에|에서|으로|로|과|와|도|만|랑|이랑"


class KnowledgeError(Exception):
    pass


class UnitError(ValueError):
    pass


@dataclass(frozen=True)
class Parameter:
    """레시피 파라미터 정의. 온톨로지에서 vd:jsonField가 달린 속성 하나."""

    iri: URIRef
    key: str  # AI 추출 JSON의 키 (속성의 로컬 이름, 예: targetTemperature)
    json_field: str  # 레시피 JSON 경로 (예: heating.target_c)
    label: str
    synonyms: tuple[str, ...]
    definition: str
    hint: str  # LLM 추출 프롬프트용 지침 (vd:extractionHint)
    value_pattern: str  # 문자열 값의 형식 정규식 (vd:valuePattern), 없으면 ""
    default_value: float | None  # 온톨로지에 적힌 기본값 (vd:defaultValue). recipe.py와 같아야 한다
    default_rationale: str  # 기본값을 그렇게 정한 이유 (vd:defaultRationale)
    kind: str  # "number" | "text" | "choice"
    unit: URIRef | None
    unit_symbol: str
    required: bool
    choices: tuple[tuple[str, str], ...] = ()  # (코드, 한국어 이름)


@dataclass(frozen=True)
class Unit:
    iri: URIRef
    symbol: str
    quantity_kind: URIRef
    multiplier: float
    offset: float


@dataclass(frozen=True)
class Entity:
    iri: URIRef
    type: URIRef
    label: str  # 한국어 대표 이름
    surface: str = ""  # 문장에서 찾은 표기


class Review(BaseModel):
    ok: bool
    issues: list[Issue]
    facts: list[str]
    similar_runs: dict


def _norm(text: str) -> str:
    return re.sub(r"\s+", "", str(text)).lower()


def _flatten(data: dict, prefix: str = "") -> dict:
    flat = {}
    for key, value in data.items():
        if isinstance(value, dict):
            flat |= _flatten(value, f"{prefix}{key}.")
        else:
            flat[prefix + key] = value
    return flat


def _korean(values) -> str | None:
    values = list(values)
    for v in values:
        if getattr(v, "language", None) == "ko":
            return str(v)
    return str(values[0]) if values else None


class KnowledgeBase:
    def __init__(
        self,
        ontology_dir: Path = ONTOLOGY_DIR,
        equipment: str = "Chamber01",
        runs_dir: Path = RUNS_DIR,
    ):
        self.graph = Graph()
        for name in KNOWLEDGE_FILES:
            self.graph.parse(ontology_dir / name)
        self.graph.bind("vd", VD)
        self.shapes = Graph().parse(ontology_dir / "vd-shapes.ttl")
        self.quality_shapes = Graph().parse(ontology_dir / "vd-data-quality.ttl")
        self.equipment = VD[equipment]
        if (self.equipment, RDF.type, VD.VacuumDryingChamber) not in self.graph:
            raise KnowledgeError(f"온톨로지에 장비 vd:{equipment}가 없습니다.")
        self.runs_dir = Path(runs_dir)
        self._runs_cache: tuple[tuple, Graph] | None = None

        self.units = self._load_units()
        self._unit_by_label = {
            _norm(label): unit
            for unit in self.units.values()
            for label in [unit.symbol, *self.graph.objects(unit.iri, SKOS.altLabel)]
        }
        self.parameters = self._load_parameters()
        self._param_by_iri = {p.iri: p for p in self.parameters}
        self.param = {p.key: p for p in self.parameters}
        self._field_order = {path: i for i, path in enumerate(field_paths())}
        self._entity_patterns = self._load_entity_patterns()

    # ------------------------------------------------------------------ 적재

    def _load_units(self) -> dict[URIRef, Unit]:
        units = {}
        for iri in set(self.graph.subjects(QUDT.conversionMultiplier, None)):
            units[iri] = Unit(
                iri=iri,
                symbol=str(self.graph.value(iri, QUDT.symbol)),
                quantity_kind=self.graph.value(iri, QUDT.hasQuantityKind),
                multiplier=float(self.graph.value(iri, QUDT.conversionMultiplier).toPython()),
                offset=float(self.graph.value(iri, QUDT.conversionOffset, default=Literal(0.0)).toPython()),
            )
        return units

    def _load_parameters(self) -> list[Parameter]:
        order = {path: i for i, path in enumerate(field_paths())}
        params = []
        for iri, json_field in self.graph.subject_objects(VD.jsonField):
            g = self.graph
            rng = g.value(iri, RDFS.range)
            unit = g.value(iri, QUDT.hasUnit)
            if rng == XSD.double:
                kind, choices = "number", ()
            elif rng == XSD.string:
                kind, choices = "text", ()
            else:
                kind = "choice"
                choices = tuple(sorted(
                    (str(g.value(ind, VD.code)), _korean(g.objects(ind, RDFS.label)))
                    for ind in g.subjects(RDF.type, rng)
                ))
            params.append(Parameter(
                iri=iri,
                key=iri.split("#")[-1],
                json_field=str(json_field),
                label=_korean(g.objects(iri, RDFS.label)),
                synonyms=tuple(str(s) for s in g.objects(iri, SKOS.altLabel)),
                definition=_korean(g.objects(iri, SKOS.definition)) or "",
                hint=_korean(g.objects(iri, VD.extractionHint)) or "",
                value_pattern=str(g.value(iri, VD.valuePattern) or ""),
                default_value=(float(v.toPython()) if (v := g.value(iri, VD.defaultValue)) is not None else None),
                default_rationale=_korean(g.objects(iri, VD.defaultRationale)) or "",
                kind=kind,
                unit=unit,
                unit_symbol=self.units[unit].symbol if unit in self.units else "",
                required=bool(g.value(iri, VD.required, default=Literal(False)).toPython()),
                choices=choices,
            ))
        return sorted(params, key=lambda p: order.get(p.json_field, 99))

    def _load_entity_patterns(self) -> list[tuple[re.Pattern, Entity]]:
        patterns = []
        for entity_type in LINKABLE_TYPES:
            for iri in self.graph.subjects(RDF.type, entity_type):
                entity = Entity(iri=iri, type=entity_type, label=self.label(iri))
                labels = {
                    str(label)
                    for prop in (SKOS.prefLabel, SKOS.altLabel, RDFS.label)
                    for label in self.graph.objects(iri, prop)
                }
                for label in labels:
                    body = r"\s*".join(re.escape(part) for part in label.split())
                    # 영문 이름에는 한국어가 띄어쓰기 없이 붙는다 (PET라, IPA를, PET기판). 한글 이름은 조사만 허용
                    latin_start = label[0].isascii() and label[0].isalnum()
                    latin_end = label[-1].isascii() and label[-1].isalnum()
                    left = r"(?<![0-9A-Za-z])" if latin_start else r"(?<![0-9A-Za-z가-힣])"
                    right = r"(?![0-9A-Za-z])" if latin_end else rf"(?=(?:{KOREAN_PARTICLES})?(?![0-9A-Za-z가-힣]))"
                    pattern = re.compile(left + body + right, re.IGNORECASE)
                    patterns.append((pattern, entity))
        return patterns

    def label(self, iri: URIRef) -> str:
        """개체의 한국어 이름."""
        return (
            _korean(self.graph.objects(iri, SKOS.prefLabel))
            or _korean(self.graph.objects(iri, RDFS.label))
            or iri.split("#")[-1]
        )

    # ------------------------------------------------------------------ 무결성·장비

    def check_integrity(self) -> list[str]:
        """지식 기반이 무결성 규칙을 지키는지 검사하고, 위반 메시지를 돌려준다."""
        conforms, results, _ = shacl_validate(
            self.graph, shacl_graph=self.quality_shapes, advanced=True, allow_warnings=True
        )
        if conforms:
            return []
        messages = []
        for r in results.subjects(RDF.type, SH.ValidationResult):
            focus = results.value(r, SH.focusNode)
            name = focus.split("#")[-1] if isinstance(focus, URIRef) else str(focus)
            messages.append(f"{name}: {_korean(results.objects(r, SH.resultMessage))}")
        return sorted(messages)

    def max_temperature(self) -> tuple[float, str]:
        """장비 최대 온도와 그 값을 정하는 구성품 이름."""
        rows = list(self.graph.query(
            """SELECT ?limit ?part WHERE {
                   ?eq vd:hasComponent ?c .
                   ?c vd:maxOperatingTemperature ?limit ; rdfs:label ?part .
                   FILTER (langMatches(lang(?part), "ko"))
               } ORDER BY ?limit LIMIT 1""",
            initBindings={"eq": self.equipment},
        ))
        if not rows:
            raise KnowledgeError("장비 구성품에 최대 사용 온도가 없습니다.")
        return float(rows[0].limit.toPython()), str(rows[0].part)

    def limits(self) -> EquipmentLimits:
        rows = list(self.graph.query(
            """SELECT ?ramp ?pump ?atm ?safe ?door ?overshoot ?maxProc ?heatExtra ?vent ?cool WHERE {
                   ?eq vd:ventedPressure ?atm ; vd:overshootMargin ?overshoot ;
                       vd:maxProcessDuration ?maxProc ; vd:heatTimeoutExtra ?heatExtra ;
                       vd:ventTimeout ?vent ; vd:coolTimeout ?cool ;
                       vd:hasComponent ?h , ?p , ?d .
                   ?h a vd:Heater ; vd:maxRampRate ?ramp .
                   ?p a vd:VacuumPump ; vd:ultimatePressure ?pump .
                   ?d a vd:AutomaticDoor ; vd:safeOpenTemperature ?safe ; vd:travelTimeout ?door .
               }""",
            initBindings={"eq": self.equipment},
        ))
        if len(rows) != 1:
            raise KnowledgeError(
                f"장비 한계값을 하나로 정할 수 없습니다 (결과 {len(rows)}개). "
                "python3 -m process_ai.knowledge check 로 확인하세요."
            )
        r = {k: float(v.toPython()) for k, v in rows[0].asdict().items()}
        return EquipmentLimits(
            max_temp_c=self.max_temperature()[0],
            max_ramp_c_per_min=r["ramp"],
            min_pressure_kpa=r["pump"],
            atm_pressure_kpa=r["atm"],
            safe_open_temp_c=r["safe"],
            overshoot_margin_c=r["overshoot"],
            max_process_min=r["maxProc"],
            door_timeout_s=r["door"],
            heat_timeout_extra_min=r["heatExtra"],
            vent_timeout_s=r["vent"],
            cool_timeout_min=r["cool"],
        )

    def component_value(self, prop: URIRef) -> float | None:
        """장비 구성품 중 prop 값을 가진 것의 값 (예: vd:ultimatePressure → 펌프 최저 압력)."""
        for component in self.graph.objects(self.equipment, VD.hasComponent):
            value = self.graph.value(component, prop)
            if value is not None:
                return float(value.toPython())
        return None

    # ------------------------------------------------------------------ 용어·단위

    def link_entities(self, text: str) -> list[Entity]:
        """문장에서 온톨로지 개체 이름을 찾는다. 겹치면 더 긴 이름을 택한다."""
        found = []
        for pattern, entity in self._entity_patterns:
            for m in pattern.finditer(text):
                found.append((m.start(), m.end(), entity))
        found.sort(key=lambda f: (-(f[1] - f[0]), f[0]))
        taken: list[tuple[int, int]] = []
        result = []
        for start, end, entity in found:
            if any(start < e and s < end for s, e in taken):
                continue
            taken.append((start, end))
            result.append((start, Entity(entity.iri, entity.type, entity.label, text[start:end])))
        seen, unique = set(), []
        for _, entity in sorted(result, key=lambda r: r[0]):
            if entity.iri not in seen:
                seen.add(entity.iri)
                unique.append(entity)
        return unique

    def link_name(self, name: str, entity_type: URIRef) -> Entity | None:
        """이름 하나를 주어진 종류의 개체와 연결한다 (예: "IPA" → 이소프로판올)."""
        if not name or not name.strip():
            return None
        for pattern, entity in self._entity_patterns:
            if entity.type == entity_type and pattern.fullmatch(name.strip()):
                return Entity(entity.iri, entity.type, entity.label, name.strip())
        matches = [e for e in self.link_entities(name) if e.type == entity_type]
        return matches[0] if len(matches) == 1 else None

    def is_generic_term(self, text: str) -> bool:
        """'시편', '기판', '없음'처럼 재료 이름이 아닌 일반 명사인지 (vd:GenericTerm)."""
        return any(e.type == VD.GenericTerm for e in self.link_entities(text) if e.surface == text.strip())

    def entity_names(self, entity_type: URIRef) -> list[str]:
        names = []
        for iri in sorted(self.graph.subjects(RDF.type, entity_type)):
            aliases = [str(a) for a in self.graph.objects(iri, SKOS.altLabel) if getattr(a, "language", None) != "ko"]
            label = self.label(iri)
            names.append(f"{label}({', '.join(sorted(aliases))})" if aliases else label)
        return names

    def compatible_units(self, param: Parameter) -> list[str]:
        """파라미터와 같은 물리량의 단위 기호들 (예: 시간 → 초, 분, 시간)."""
        kind = self.units[param.unit].quantity_kind
        return [u.symbol for u in self.units.values() if u.quantity_kind == kind]

    def to_canonical(self, param: Parameter, value: float, unit_text: str) -> tuple[float, str | None]:
        """값을 파라미터의 기준 단위로 바꾼다. 바꿨으면 설명 문구도 돌려준다."""
        target = self.units[param.unit]
        if not unit_text or not unit_text.strip():
            return value, f"{param.label}: 단위가 없어 {target.symbol}로 해석했습니다."
        unit = self._unit_by_label.get(_norm(unit_text))
        if unit is None:
            raise UnitError(f"{param.label}의 단위 '{unit_text}'를 알 수 없습니다.")
        if unit.quantity_kind != target.quantity_kind:
            raise UnitError(f"'{unit_text}'는 {param.label}에 쓸 수 없는 단위입니다.")
        if unit.iri == target.iri:
            return value, None
        converted = (value + unit.offset) * unit.multiplier / target.multiplier - target.offset
        converted = float(f"{converted:.4g}")
        return converted, f"{param.label}: {value:g} {unit.symbol} → {converted:g} {target.symbol} (단위 변환)"

    # ------------------------------------------------------------------ 레시피

    def recipe_graph(self, recipe: Recipe, node: URIRef = VD.DraftRecipe) -> Graph:
        g = Graph()
        g.add((node, RDF.type, VD.Recipe))
        g.add((node, VD.forEquipment, self.equipment))
        values = _flatten(recipe.model_dump())
        for p in self.parameters:
            value = values.get(p.json_field)
            if value is None or value == "":
                continue
            if p.kind == "number":
                g.add((node, p.iri, Literal(float(value), datatype=XSD.double)))
            elif p.kind == "choice":
                for ind in self.graph.subjects(VD.code, Literal(value)):
                    g.add((node, p.iri, ind))
            else:
                g.add((node, p.iri, Literal(value)))
        for name, entity_type, prop in (
            (recipe.solvent, VD.Solvent, VD.usesSolvent),
            (recipe.substrate, VD.Substrate, VD.usesSubstrate),
        ):
            if entity := self.link_name(name, entity_type):
                g.add((node, prop, entity.iri))
        return g

    def validate(self, recipe: Recipe) -> list[Issue]:
        data = Graph()
        data += self.graph
        data += self.recipe_graph(recipe)
        _, results, _ = shacl_validate(data, shacl_graph=self.shapes, advanced=True, allow_warnings=True)
        issues, seen = [], set()
        for r in results.subjects(RDF.type, SH.ValidationResult):
            path = results.value(r, SH.resultPath)
            param = self._param_by_iri.get(path)
            message = _korean(results.objects(r, SH.resultMessage)) or "규칙 위반"
            issue = Issue(
                level=SEVERITY[results.value(r, SH.resultSeverity)],
                field=param.json_field if param else "recipe",
                message=re.sub(r"(?<=\d)\.0(?!\d)", "", message),  # 150.0°C → 150°C
            )
            key = (issue.level, issue.field, issue.message)
            if key not in seen:
                seen.add(key)
                issues.append(issue)
        return sorted(issues, key=lambda i: (LEVEL_ORDER[i.level], self._field_order.get(i.field, 99)))

    def boiling_point_at(self, solvent: URIRef, pressure_kpa: float) -> float | None:
        """Antoine 식으로 추정한 압력별 끓는점 (°C)."""
        a, b, c = (self.graph.value(solvent, p) for p in (VD.antoineA, VD.antoineB, VD.antoineC))
        if None in (a, b, c) or pressure_kpa <= 0:
            return None
        denominator = float(a) - math.log10(pressure_kpa * MMHG_PER_KPA)
        if denominator <= 0:
            return None
        return float(b) / denominator - float(c)

    def entity_facts(self, entity: Entity, pressure_kpa: float | None = None) -> list[str]:
        """재료 하나에 대한 온톨로지 지식을 문장으로 정리한다."""
        g = self.graph
        if entity.type == VD.Solvent:
            details = [f"끓는점 {float(g.value(entity.iri, VD.boilingPoint)):g}°C"]
            if (flash := g.value(entity.iri, VD.flashPoint)) is not None:
                details.append(f"인화점 {float(flash):g}°C")
            facts = [f"{entity.label} (CAS {g.value(entity.iri, VD.casNumber)}): {', '.join(details)}"]
            if pressure_kpa is not None and (boiling := self.boiling_point_at(entity.iri, pressure_kpa)) is not None:
                facts.append(f"{pressure_kpa:g} kPa에서 {entity.label}의 끓는점은 약 {boiling:.0f}°C입니다 (Antoine 식 추정).")
            return facts
        if entity.type == VD.Substrate:
            return [f"{entity.label}의 최대 사용 온도는 {float(g.value(entity.iri, VD.maxOperatingTemperature)):g}°C입니다."]
        if entity.type == VD.Coating:
            return [f"{entity.label}은(는) 기판 위에 올린 코팅 재료입니다. 레시피의 기판·용매 항목에는 들어가지 않습니다."]
        return []

    def facts(self, recipe: Recipe) -> list[str]:
        """레시피와 관련된 온톨로지 지식을 문장으로 정리한다."""
        limit, part = self.max_temperature()
        facts = [f"이 장비의 최대 온도는 {limit:g}°C입니다 ({part} 기준)."]
        if solvent := self.link_name(recipe.solvent, VD.Solvent):
            facts += self.entity_facts(solvent, recipe.vacuum.target_kpa)
        if substrate := self.link_name(recipe.substrate, VD.Substrate):
            facts += self.entity_facts(substrate)
        return facts

    def review(self, recipe: Recipe) -> Review:
        issues = self.validate(recipe)
        return Review(
            ok=not any(i.level == "error" for i in issues),
            issues=issues,
            facts=self.facts(recipe),
            similar_runs=self.similar_runs(recipe),
        )

    # ------------------------------------------------------------------ 실행 이력

    def runs_graph(self) -> Graph:
        """runs/의 실행 기록을 PROV-O 지식그래프로 바꾼다. 기록이 바뀌지 않았으면 캐시를 쓴다."""
        summaries = sorted(self.runs_dir.glob("*/summary.json")) if self.runs_dir.exists() else []
        key = tuple((s.parent.name, s.stat().st_mtime) for s in summaries)
        if self._runs_cache and self._runs_cache[0] == key:
            return self._runs_cache[1]
        g = Graph()
        g.bind("vd", VD)
        g.bind("prov", PROV)
        for summary_path in summaries:
            run_dir = summary_path.parent
            try:
                meta = json.loads((run_dir / "recipe.json").read_text(encoding="utf-8"))
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                recipe = Recipe.model_validate(meta["recipe"])
            except (OSError, ValueError, KeyError):
                continue
            run = VD[f"run/{quote(run_dir.name)}"]
            recipe_node = VD[f"recipe/{quote(run_dir.name)}"]
            g += self.recipe_graph(recipe, recipe_node)
            g.add((run, RDF.type, VD.ProcessRun))
            g.add((run, VD.runId, Literal(run_dir.name)))
            g.add((run, PROV.used, recipe_node))
            g.add((run, VD.result, Literal(summary["result"])))
            g.add((run, PROV.startedAtTime, Literal(summary["started_at"], datatype=XSD.dateTime)))
            g.add((run, PROV.endedAtTime, Literal(summary["ended_at"], datatype=XSD.dateTime)))
            for key_name, prop in (
                ("drying_min", VD.dryingDuration),
                ("process_min", VD.processDuration),
                ("max_temp_c", VD.maxObservedTemperature),
                ("min_pressure_kpa", VD.minObservedPressure),
            ):
                if summary.get(key_name) is not None:
                    g.add((run, prop, Literal(float(summary[key_name]), datatype=XSD.double)))
            if summary.get("endpoint_detected") is not None:
                g.add((run, VD.endpointDetected, Literal(bool(summary["endpoint_detected"]))))
            if summary.get("reason"):
                g.add((run, VD.abortReason, Literal(summary["reason"])))
            if operator := meta.get("approved_by"):
                agent = VD[f"operator/{quote(operator)}"]
                g.add((agent, RDF.type, VD.Operator))
                g.add((agent, RDFS.label, Literal(operator)))
                g.add((run, PROV.wasAssociatedWith, agent))
        self._runs_cache = (key, g)
        return g

    def similar_runs(self, recipe: Recipe, limit: int = 5) -> dict:
        """같은 용매, 목표 온도 ±15°C인 과거 실행을 찾는다."""
        runs = self.runs_graph()
        solvent = self.link_name(recipe.solvent, VD.Solvent)
        if solvent:
            match, bindings = "?r vd:usesSolvent ?solvent .", {"solvent": solvent.iri}
        elif recipe.solvent.strip():
            match = "?r vd:solventName ?name . FILTER (LCASE(STR(?name)) = LCASE(STR(?wanted)))"
            bindings = {"wanted": Literal(recipe.solvent.strip())}
        else:
            match, bindings = "FILTER NOT EXISTS { ?r vd:solventName ?any }", {}
        rows = list(runs.query(
            f"""SELECT ?id ?start ?temp ?press ?result ?detected ?drying WHERE {{
                    ?run a vd:ProcessRun ; vd:runId ?id ; prov:startedAtTime ?start ;
                         prov:used ?r ; vd:result ?result .
                    ?r vd:targetTemperature ?temp ; vd:targetPressure ?press .
                    OPTIONAL {{ ?run vd:endpointDetected ?detected }}
                    OPTIONAL {{ ?run vd:dryingDuration ?drying }}
                    {match}
                    FILTER (ABS(?temp - ?target) <= {SIMILAR_TEMP_RANGE_C})
                }} ORDER BY DESC(?start)""",
            initNs={"vd": VD, "prov": PROV},
            initBindings={"target": Literal(recipe.heating.target_c, datatype=XSD.double)} | bindings,
        ))
        items = [{
            "run_id": str(r.id),
            "started_at": str(r.start),
            "target_c": float(r.temp.toPython()),
            "target_kpa": float(r.press.toPython()),
            "result": str(r.result),
            "endpoint_detected": r.detected.toPython() if r.detected is not None else None,
            "drying_min": float(r.drying.toPython()) if r.drying is not None else None,
        } for r in rows]
        if not items:
            return {"count": 0, "summary": None, "runs": []}
        completed = [i for i in items if i["result"] == "COMPLETE"]
        detected = [i for i in completed if i["endpoint_detected"] and i["drying_min"] is not None]
        summary = f"조건이 비슷한 과거 실험 {len(items)}건 (같은 용매, 목표 온도 ±{SIMILAR_TEMP_RANGE_C:g}°C): 완료 {len(completed)}건"
        if detected:
            average = sum(i["drying_min"] for i in detected) / len(detected)
            summary += f", 종료점 검출 {len(detected)}건, 평균 건조 {average:.1f}분"
        return {"count": len(items), "summary": summary, "runs": items[:limit]}

    def export_turtle(self) -> str:
        """온톨로지와 실행 이력을 합친 지식그래프 (Protégé, GraphDB, Neo4j n10s로 열 수 있음)."""
        g = Graph()
        g += self.graph
        g += self.runs_graph()
        for prefix, ns in self.graph.namespaces():
            g.bind(prefix, ns)
        g.bind("prov", PROV)
        return g.serialize(format="turtle")

    def materials(self) -> dict:
        return {
            "solvents": sorted(self.label(i) for i in self.graph.subjects(RDF.type, VD.Solvent)),
            "substrates": sorted(self.label(i) for i in self.graph.subjects(RDF.type, VD.Substrate)),
            "coatings": sorted(self.label(i) for i in self.graph.subjects(RDF.type, VD.Coating)),
        }


def main(argv: list[str]) -> int:
    kb = KnowledgeBase()
    command = argv[0] if argv else "check"
    if command == "check":
        problems = kb.check_integrity()
        for problem in problems:
            print("✗", problem)
        print("무결성 검사 통과" if not problems else f"문제 {len(problems)}건")
        return 1 if problems else 0
    if command == "limits":
        for name, value in kb.limits().model_dump().items():
            print(f"{name:24s} {value:g}")
        return 0
    if command == "export":
        text = kb.export_turtle()
        if len(argv) > 1:
            Path(argv[1]).write_text(text, encoding="utf-8")
        else:
            print(text)
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
