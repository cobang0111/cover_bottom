# 제약 조건 생성하고 optimize-mask로 제약 조건을 넘기는 코드
from typing import Dict, Any, List, Set, Tuple
from collections import defaultdict

# 기존 규칙/좌표 변환 유틸 재사용
from app.services.cad_service import HORIZONTAL_RULES, VERTICAL_RULES, _px_to_cad, _cad_to_px

def _is_h_up_or_down(rid: str) -> bool:
    return rid.endswith('_up') or rid.endswith('_down')

def _h_base(rid: str):
    if _is_h_up_or_down(rid):
        return rid.rsplit('_', 1)[0]
    return None

def analyze_from_rects(
    required_rects: List[Any],
    forbidden_rects: List[Any],
    cad_xlim: Tuple[float, float],
    cad_ylim: Tuple[float, float],
    pw: int,
    ph: int
) -> Dict[str, Any]:
    # baseline = 외곽 bbox의 min_y와 동일하다고 가정 (cad_service에서 outline rect를 추가한 뒤 ylim을 읽음)
    baseline = cad_ylim[0]

    # h5 전용 오프셋 (붙여넣은 코드와 동일)
    H5_Y_OFFSETS = {
        'h5_1': (28, 80), 'h5_6': (28, 80), 'h5_2': (22, 80),
        'h5_5': (22, 80), 'h5_3': (15, 45), 'h5_4': (15, 45),
    }

    # 1) 수평 규칙을 모두 밴드 사각형으로 구성 (붙여넣은 코드의 _build_rule_rects 로직)
    def _build_horizontal_rect_rules():
        # 고정 정의 + 밴드에 포함
        h1_p_rule = {'id': 'h1_p', 'y_range': (255.0, 270.0), 'x_range': (-570.0, 570.0)}
        rev_forming_rule_def = {'id': 'rev_forming', 'x_range': (-535.0, 535.0), 'y_range': (200.58, 225.58)}

        y_groups = defaultdict(list)
        h5_rules_lines = []

        # y-밴드용 중심값
        h1_p_y_center = (h1_p_rule['y_range'][0] + h1_p_rule['y_range'][1]) / 2.0
        rev_forming_y_center = (rev_forming_rule_def['y_range'][0] + rev_forming_rule_def['y_range'][1]) / 2.0

        y_groups[h1_p_y_center].append(h1_p_rule)
        y_groups[rev_forming_y_center].append(rev_forming_rule_def)

        # HORIZONTAL_RULES에서 h5_*는 따로 처리, 나머지는 y그룹에 담음
        for rule in HORIZONTAL_RULES:
            rid = rule.get('id')
            if not rid or 'start_point' not in rule:
                continue
            if rid.startswith('h5'):
                h5_rules_lines.append(rule)
            else:
                y_coord = rule['start_point'][1]
                y_groups[y_coord].append(rule)

        # y 레벨 정렬(내림차순) 후, 인접 레벨 중간선을 이용해 각 레벨의 밴드 범위 계산
        sorted_y = sorted(y_groups.keys(), reverse=True)
        y_ranges: Dict[float, Tuple[float, float]] = {}
        for i, y_current in enumerate(sorted_y):
            if i == 0:
                y_next = sorted_y[i + 1]
                y_lower = (y_current + y_next) / 2.0
                y_upper = h1_p_rule['y_range'][1]
            else:
                y_previous = sorted_y[i - 1]
                y_upper = (y_current + y_previous) / 2.0

            if i == len(sorted_y) - 1:
                y_previous = sorted_y[i - 1]
                mid_boundary = (y_current + y_previous) / 2.0
                delta = mid_boundary - y_current
                y_lower = y_current - delta
            elif i != 0:
                y_next = sorted_y[i + 1]
                y_lower = (y_current + y_next) / 2.0

            y_ranges[y_current] = (y_lower, y_upper)

        # 실제 수평 사각형 규칙 생성
        horizontal_rect_rules: List[Dict[str, Any]] = []
        for y_coord, rules in y_groups.items():
            for rule in rules:
                rid = rule['id']
                if 'x_range' in rule:
                    x_range = rule['x_range']
                else:
                    p1 = rule['start_point']; p2 = rule['end_point']
                    x_center = (p1[0] + p2[0]) / 2.0
                    x_range = (x_center - 20.0, x_center + 20.0)
                horizontal_rect_rules.append({
                    'id': rid,
                    'x_range': x_range,
                    'y_range': y_ranges[y_coord]
                })

        # h5_*: baseline 기반 오프셋 적용
        for rule in h5_rules_lines:
            rid = rule['id']
            p1 = rule['start_point']; p2 = rule['end_point']
            x_center = (p1[0] + p2[0]) / 2.0
            off_lo, off_hi = H5_Y_OFFSETS.get(rid, (0.0, 0.0))
            horizontal_rect_rules.append({
                'id': rid,
                'x_range': (x_center - 20.0, x_center + 20.0),
                'y_range': (baseline + off_lo, baseline + off_hi)
            })

        return horizontal_rect_rules

    horizontal_rect_rules = _build_horizontal_rect_rules()
    vertical_rect_rules = VERTICAL_RULES  # 그대로 사용

    def rect_px_to_cad_box(r) -> Dict[str, Tuple[float, float]]:
        x1, y1 = _px_to_cad(r.x1, r.y1, cad_xlim, cad_ylim, pw, ph)
        x2, y2 = _px_to_cad(r.x2, r.y2, cad_xlim, cad_ylim, pw, ph)
        return {'x': tuple(sorted((x1, x2))), 'y': tuple(sorted((y1, y2)))}

    # 수집 버킷
    wide_found_h: Set[str] = set()
    wide_found_v: Set[str] = set()
    final_non_zero_from_tall: Set[str] = set()

    # tall 박스에서 작은쪽 v*_p -> fixed_zero 보강
    final_fixed_zero_from_tall_p: Set[str] = set()
    # forbidden 박스 CAD 좌표 수집
    forbidden_boxes: List[Dict[str, Tuple[float, float]]] = []

    # required 분석
    for r in required_rects:
        is_tall = abs(r.y2 - r.y1) >= abs(r.x2 - r.x1)
        u = rect_px_to_cad_box(r)

        found_h = {rr['id'] for rr in horizontal_rect_rules if
                   (u['x'][0] < rr['x_range'][1] and u['x'][1] > rr['x_range'][0] and
                    u['y'][0] < rr['y_range'][1] and u['y'][1] > rr['y_range'][0])}
        found_v = {vr['id'] for vr in vertical_rect_rules if
                   (u['x'][0] < sorted(vr['x_range'])[1] and u['x'][1] > sorted(vr['x_range'])[0] and
                    u['y'][0] < sorted(vr['y_range'])[1] and u['y'][1] > sorted(vr['y_range'])[0])}

        if is_tall:
            # (v_i, v_{i+1}) 동시 검출 시 v_{i+1} 제거
            non_zero_candidates = set(found_v)
            for layer in range(1, 5):
                for i in range(1, 6):
                    v_small = f"v{i}_{layer}"
                    v_large = f"v{i+1}_{layer}"
                    if {v_small, v_large}.issubset(found_v):
                        if v_large in non_zero_candidates:
                            non_zero_candidates.remove(v_large)
            final_non_zero_from_tall.update(non_zero_candidates)
            pairs_to_check_for_p_zero = [
                ('v3_1', 'v4_1'),
                ('v3_2', 'v4_2'),
                ('v3_3', 'v4_3'),
            ]
            for v_small, v_large in pairs_to_check_for_p_zero:
                if {v_small, v_large}.issubset(found_v):
                    final_fixed_zero_from_tall_p.add(f"{v_small}_p")
        else:
            wide_found_h.update(found_h)
            wide_found_v.update(found_v)

    # forbidden 분석
    all_h_forb, all_v_forb = set(), set()
    for r in forbidden_rects:
        u = rect_px_to_cad_box(r)
        found_h = {rr['id'] for rr in horizontal_rect_rules if
                   (u['x'][0] < rr['x_range'][1] and u['x'][1] > rr['x_range'][0] and
                    u['y'][0] < rr['y_range'][1] and u['y'][1] > rr['y_range'][0])}
        found_v = {vr['id'] for vr in vertical_rect_rules if
                   (u['x'][0] < sorted(vr['x_range'])[1] and u['x'][1] > sorted(vr['x_range'])[0] and
                    u['y'][0] < sorted(vr['y_range'])[1] and u['y'][1] > sorted(vr['y_range'])[0])}
        all_h_forb.update(found_h)
        all_v_forb.update(found_v)
        forbidden_boxes.append({
            "x1": u["x"][0],
            "x2": u["x"][1],
            "y1": u["y"][0],
            "y2": u["y"][1],
        })

    # 조건 조립 (붙여넣은 코드의 최종 JSON과 동일한 규칙)
    add_unconditional: List[List[str]] = []  # 사용하지 않음(빈 배열 유지)
    must_equal: List[List[str]] = []
    not_equal: List[List[str]] = []
    upper_bound: Dict[str, Any] = {}
    fixed_value: Dict[str, Any] = {}
    fixed_zero = sorted(list(all_v_forb))
    non_zero = sorted(list(final_non_zero_from_tall))  # wide v는 포함하지 않음

    # required: 동일 base에서 up/down이 함께 검출되면 not_equal
    bases = defaultdict(set)
    for hid in wide_found_h:
        if _is_h_up_or_down(hid):
            base, suffix = hid.rsplit('_', 1)
            bases[base].add(suffix)
    for base, suffixes in bases.items():
        if {'up', 'down'}.issubset(suffixes):
            not_equal.append([f"{base}_up", f"{base}_down"])

    # required: 이웃 레이어 등식(h{a,b}_up == h{a-1,b}_down)
    def _parse_h_id(hid: str):
        parts, suffix = hid.replace('h', '').rsplit('_', 1)
        a, b = parts.split('_')
        return int(a), int(b), suffix

    '''
    seen_pairs = set()
    for hid in wide_found_h:
        if not (hid.startswith('h') and _is_h_up_or_down(hid)):
            continue
        a, b, suffix = _parse_h_id(hid)
        if suffix == 'up' and a > 1:
            prev_down = f"h{a-1}_{b}_down"
            pair = tuple(sorted([hid, prev_down]))
            if prev_down in wide_found_h and pair not in seen_pairs:
                must_equal.append([hid, prev_down])
                seen_pairs.add(pair)
    '''
    seen_pairs = set()
    exception_pairs_for_fixed_zero = {
        ('h2_1_down', 'h3_1_up'),
        ('h2_2_down', 'h3_2_up'),
        ('h2_3_down', 'h3_3_up'),
        ('h2_4_down', 'h3_4_up'),
        ('h2_5_down', 'h3_5_up'),
    }
    for hid in wide_found_h:
        if not (hid.startswith('h') and _is_h_up_or_down(hid)):
            continue
        a, b, suffix = _parse_h_id(hid)

        if a == 2 and suffix == 'up':
            target_h2_down = f"h2_{b}_down"
            upper_bound[target_h2_down] = 70.0

        if suffix == 'up' and a > 1:
            prev_down = f"h{a-1}_{b}_down"
            pair = tuple(sorted([hid, prev_down]))
            if prev_down in wide_found_h and pair not in seen_pairs:
                if pair in exception_pairs_for_fixed_zero:
                    fixed_zero.extend([prev_down, hid])
                else:
                    must_equal.append([hid, prev_down])
                seen_pairs.add(pair)

    # forbidden: base 단위 up/down 동일 강제(must_equal), 수직은 fixed_zero
    processed = set()
    for hid in sorted(list(all_h_forb)):
        base = _h_base(hid)
        if base and base not in processed:
            must_equal.append([f"{base}_up", f"{base}_down"])
            processed.add(base)

    # forbidden: 특정 v*\_p는 567(int)로 고정
    v_rules_for_fixed_value = {'v5_3', 'v5_4', 'v6_3', 'v6_4', 'v1_3', 'v1_4'}
    for vid in set(all_v_forb):
        if vid in v_rules_for_fixed_value:
            fixed_value[f"{vid}_p"] = 567  # int로 맞춤

    p_alignment_pairs = [
        ('v5_3', 'v5_4'),
        ('v2_3', 'v2_4')
    ]
    for v_a, v_b in p_alignment_pairs:
        # all_v_forb는 위에서 수집한 금지구역 내 수직 규칙 집합입니다.
        if v_a in all_v_forb and v_b in all_v_forb:
            p_a = f"{v_a}_p"
            p_b = f"{v_b}_p"
            must_equal.append([p_a, p_b])
            
    # 최종 정리: not_equal vs must_equal 충돌 제거, non_zero에서 fixed_zero 제거, 중복 제거/정렬
    def _pair_key(pair: List[str]):
        a, b = sorted(pair)
        base_a = _h_base(a) if _is_h_up_or_down(a) else a
        base_b = _h_base(b) if _is_h_up_or_down(b) else b
        return base_a if base_a and base_a == base_b else (a, b)

    must_eq_keys = set(_pair_key(p) for p in must_equal)
    not_equal = [p for p in not_equal if _pair_key(p) not in must_eq_keys]

    fixed_zero_set = set(fixed_zero)
    non_zero = [v for v in non_zero if v not in fixed_zero_set]

    # 정렬 및 유니크 처리
    def _uniq_sorted_pairs(pairs: List[List[str]]) -> List[List[str]]:
        unique_pairs = list(set(tuple(sorted(p)) for p in pairs))
        return sorted([list(p) for p in unique_pairs])

    fixed_zero = sorted(list(set(fixed_zero)))
    non_zero = sorted(list(set(non_zero)))
    not_equal = _uniq_sorted_pairs(not_equal)
    must_equal = _uniq_sorted_pairs(must_equal)

    fixed_zero.extend(list(final_fixed_zero_from_tall_p))

    print(f"upper_bound: {upper_bound}")

    return {
        'add_unconditional': [],
        'fixed_zero': fixed_zero,
        'non_zero': non_zero,
        'not_equal': not_equal,
        'fixed_value': fixed_value,
        'upper_bound': upper_bound,
        'must_equal': must_equal,
        'forbidden_zones': forbidden_boxes,
    }