"""Mini NPU Simulator

입력 패턴과 필터(Cross / X)를 위치별로 곱해 더하는 MAC 연산으로
패턴을 판정하고, 크기별 연산 시간을 측정한다.

- 모드 1: 3x3 필터 A/B와 패턴을 콘솔에서 직접 입력
- 모드 2: data.json의 필터/패턴을 일괄 분석 (PASS/FAIL + 요약)
- 모드 3: [보너스] 크기 N을 입력받아 Cross / X 패턴 생성 + 판정 + 성능 측정

외부 라이브러리 없이 표준 라이브러리(json, math, os, re, sys, time)만 사용한다.
"""

import json
import math
import os
import re
import sys
import time
import unicodedata

# ---------------------------------------------------------------------------
# 설정값
# ---------------------------------------------------------------------------
EPSILON = 1e-9               # 이 값보다 점수 차이가 작으면 동점으로 본다
REPEAT = 10                  # 성능 측정 반복 횟수
PERF_SIZES = (3, 5, 13, 25)  # 성능 분석 대상 크기
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.json")

# 표준 라벨
CROSS = "Cross"
X = "X"
UNDECIDED = "UNDECIDED"

# 데이터에 등장하는 다양한 표기 -> 표준 라벨
LABEL_ALIASES = {
    "+": CROSS,
    "cross": CROSS,
    "x": X,
}

LINE = "#" + "-" * 40


# ---------------------------------------------------------------------------
# 1. 데이터 구조: n x n 2차원 배열
# ---------------------------------------------------------------------------
def create_matrix(n, fill=0.0):
    """n x n 크기의 2차원 배열을 만든다. (행마다 별도의 리스트)"""
    return [[fill] * n for _ in range(n)]


def get_value(matrix, row, col):
    """특정 위치의 값을 읽는다."""
    return matrix[row][col]


def set_value(matrix, row, col, value):
    """특정 위치에 값을 저장한다."""
    matrix[row][col] = value


def validate_matrix(matrix, expected_size=None):
    """정사각형 숫자 배열인지 검사한다.

    반환: (크기, 오류 메시지) - 정상이면 오류 메시지는 None
    """
    if not isinstance(matrix, list) or len(matrix) == 0:
        return None, "배열이 비어 있거나 2차원 리스트가 아닙니다"

    rows = len(matrix)
    for r, row in enumerate(matrix):
        if not isinstance(row, list):
            return None, "{}행이 리스트가 아닙니다".format(r + 1)
        if len(row) != rows:
            return None, "{}행의 열 수({})가 행 수({})와 다릅니다".format(r + 1, len(row), rows)
        for value in row:
            # bool은 int의 하위 타입이라 별도로 걸러낸다
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return None, "{}행에 숫자가 아닌 값({!r})이 있습니다".format(r + 1, value)

    if expected_size is not None and rows != expected_size:
        return None, "크기 불일치: 패턴 {0}x{0}, 필터 {1}x{1}".format(rows, expected_size)
    return rows, None


def to_float_matrix(matrix):
    """모든 값을 float로 변환한 새 배열을 만든다."""
    n = len(matrix)
    result = create_matrix(n)
    for r in range(n):
        for c in range(n):
            set_value(result, r, c, float(get_value(matrix, r, c)))
    return result


# ---------------------------------------------------------------------------
# 2. 라벨 정규화
# ---------------------------------------------------------------------------
def normalize_label(raw):
    """'+', 'cross', 'x' 등 다양한 표기를 표준 라벨(Cross / X)로 바꾼다.

    해석할 수 없으면 None을 반환한다.
    """
    if not isinstance(raw, str):
        return None
    return LABEL_ALIASES.get(raw.strip().lower())


# ---------------------------------------------------------------------------
# 3. MAC 연산 (반복문 직접 구현)
# ---------------------------------------------------------------------------
def mac(pattern, filt):
    """같은 위치끼리 곱한 뒤 모두 더한다. (N x N -> 곱셈 N^2번)"""
    n = len(pattern)
    total = 0.0
    for r in range(n):
        for c in range(n):
            total += pattern[r][c] * filt[r][c]
    return total


def flatten(matrix):
    """[보너스] 2차원 배열을 길이 N^2의 1차원 배열로 변환한다."""
    flat = []
    for row in matrix:
        for value in row:
            flat.append(value)
    return flat


def mac_flat(pattern_flat, filter_flat):
    """[보너스] 1차원 배열 버전 MAC 연산."""
    total = 0.0
    for i in range(len(pattern_flat)):
        total += pattern_flat[i] * filter_flat[i]
    return total


# ---------------------------------------------------------------------------
# 4. 점수 비교 (epsilon 기반)
# ---------------------------------------------------------------------------
def judge(score_a, score_b, label_a, label_b):
    """두 점수를 비교해 더 큰 쪽 라벨을 반환한다.

    |score_a - score_b| < EPSILON 이면 부동소수점 오차 범위의 동점으로 보고
    UNDECIDED를 반환한다.
    """
    if abs(score_a - score_b) < EPSILON:
        return UNDECIDED
    return label_a if score_a > score_b else label_b


# ---------------------------------------------------------------------------
# 5. 성능 측정
# ---------------------------------------------------------------------------
def measure_avg_ms(func, *args):
    """func(*args)를 REPEAT회 호출해 1회 평균 시간(ms)을 반환한다.

    입출력은 포함하지 않고 연산 함수 호출 구간만 측정한다.
    첫 호출(워밍업)은 측정에서 제외한다.
    """
    func(*args)
    total = 0.0
    for _ in range(REPEAT):
        start = time.perf_counter()
        func(*args)
        total += time.perf_counter() - start
    return total / REPEAT * 1000.0


# ---------------------------------------------------------------------------
# 6. [보너스] 패턴 생성기
# ---------------------------------------------------------------------------
def generate_cross(n):
    """가운데 행과 가운데 열이 1인 N x N 십자가 패턴."""
    matrix = create_matrix(n)
    mid = n // 2
    for i in range(n):
        set_value(matrix, mid, i, 1.0)
        set_value(matrix, i, mid, 1.0)
    return matrix


def generate_x(n):
    """두 대각선이 1인 N x N X 패턴."""
    matrix = create_matrix(n)
    for i in range(n):
        set_value(matrix, i, i, 1.0)
        set_value(matrix, i, n - 1 - i, 1.0)
    return matrix


# 표준 라벨 -> 생성 함수 (모드 1 자동 입력, 모드 3에서 사용)
GENERATORS = {CROSS: generate_cross, X: generate_x}


# ---------------------------------------------------------------------------
# 출력 도우미
# ---------------------------------------------------------------------------
def print_section(title):
    print()
    print(LINE)
    print("# " + title)
    print(LINE)


def format_number(value):
    """정수로 떨어지면 1.0처럼, 아니면 그대로 출력한다."""
    return repr(float(value))


def print_matrix(matrix, indent="  "):
    for row in matrix:
        print(indent + " ".join("{:g}".format(v) for v in row))


def display_width(text):
    """한글처럼 콘솔에서 2칸을 차지하는 문자를 고려한 표시 폭."""
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text)


def pad(text, width, align="right"):
    """표시 폭 기준으로 공백을 채운다."""
    space = " " * max(0, width - display_width(text))
    return text + space if align == "left" else space + text


def print_table(headers, widths, rows):
    """첫 열은 왼쪽, 나머지는 오른쪽 정렬로 표를 출력한다."""
    def render(cells):
        return "".join(pad(cell, w, "left" if i == 0 else "right")
                       for i, (cell, w) in enumerate(zip(cells, widths)))
    print(render(headers))
    print("-" * sum(widths))
    for row in rows:
        print(render(row))


def print_perf_table(rows):
    """rows: [(N, 평균 ms), ...]"""
    table = []
    for n, avg_ms in rows:
        ops = n * n
        table.append(["{0}x{0}".format(n), "{:.5f}".format(avg_ms), str(ops),
                      "{:.2f}".format(avg_ms * 1e6 / ops)])
    print_table(["크기", "평균 시간(ms)", "연산 횟수(N²)", "연산당 시간(ns)"],
                [8, 16, 16, 18], table)


# ---------------------------------------------------------------------------
# 모드 1: 사용자 입력 (3x3)
# ---------------------------------------------------------------------------
def read_line(prompt=""):
    try:
        return input(prompt)
    except EOFError:
        print("\n입력이 종료되어 프로그램을 마칩니다.")
        sys.exit(0)


def parse_row(line, n):
    """한 줄을 n개의 실수로 변환한다. 형식이 틀리면 ValueError."""
    tokens = line.split()
    if len(tokens) != n:
        raise ValueError(
            "입력 형식 오류: 각 줄에 {}개의 숫자를 공백으로 구분해 입력하세요. "
            "(입력된 개수: {})".format(n, len(tokens))
        )
    values = []
    for token in tokens:
        try:
            value = float(token)
        except ValueError:
            raise ValueError(
                "입력 형식 오류: '{}'는 숫자가 아닙니다. "
                "각 줄에 {}개의 숫자를 공백으로 구분해 입력하세요.".format(token, n)
            )
        if not math.isfinite(value):
            raise ValueError("입력 형식 오류: '{}'는 사용할 수 없는 값입니다.".format(token))
        values.append(value)
    return values


def read_matrix(title, n):
    """n줄을 입력받아 n x n 배열로 저장한다. 틀린 줄은 다시 입력받는다.

    [보너스] 첫 줄에 'cross'/'+' 또는 'x'를 입력하면 패턴 생성기로 채운다.
    """
    print("{} ({}줄 입력, 공백 구분 / 'cross' 또는 'x' 입력 시 자동 생성)".format(title, n))
    matrix = create_matrix(n)
    row = 0
    while row < n:
        line = read_line()
        if row == 0:
            label = normalize_label(line)
            if label is not None:
                matrix = GENERATORS[label](n)
                print("  -> 패턴 생성기로 {} 패턴을 채웠습니다.".format(label))
                print_matrix(matrix, indent="     ")
                return matrix
        try:
            values = parse_row(line, n)
        except ValueError as error:
            print("  " + str(error))
            print("  -> {}번째 줄을 다시 입력하세요.".format(row + 1))
            continue
        for col, value in enumerate(values):
            set_value(matrix, row, col, value)
        row += 1
    return matrix


def run_user_mode():
    n = 3

    print_section("[1] 필터 입력")
    filter_a = read_matrix("필터 A", n)
    print()
    filter_b = read_matrix("필터 B", n)

    print()
    print("✓ 필터 저장 완료")
    print("필터 A:")
    print_matrix(filter_a)
    print("필터 B:")
    print_matrix(filter_b)

    print_section("[2] 패턴 입력")
    pattern = read_matrix("패턴", n)

    print_section("[3] MAC 결과")
    score_a = mac(pattern, filter_a)
    score_b = mac(pattern, filter_b)
    # 판정 1회 = 필터 A, B 각각 MAC 1번씩
    elapsed = measure_avg_ms(lambda: (mac(pattern, filter_a), mac(pattern, filter_b)))
    verdict = judge(score_a, score_b, "A", "B")

    print("A 점수: " + format_number(score_a))
    print("B 점수: " + format_number(score_b))
    print("연산 시간(평균/{}회, A+B): {:.5f} ms".format(REPEAT, elapsed))
    if verdict == UNDECIDED:
        print("판정: 판정 불가 (|A-B| < {:g})".format(EPSILON))
    else:
        print("판정: " + verdict)

    print_section("[4] 성능 분석 (평균/{}회, MAC 1회 기준)".format(REPEAT))
    print_perf_table([(n, measure_avg_ms(mac, pattern, filter_a))])


# ---------------------------------------------------------------------------
# 모드 2: data.json 분석
# ---------------------------------------------------------------------------
def load_json(path):
    try:
        with open(path, encoding="utf-8") as file:
            return json.load(file), None
    except FileNotFoundError:
        return None, "파일을 찾을 수 없습니다: {}".format(path)
    except json.JSONDecodeError as error:
        return None, "JSON 형식 오류: {}".format(error)
    except OSError as error:
        return None, "파일을 읽을 수 없습니다: {}".format(error)


def size_sort_key(key):
    """'size_13_2' -> (13, 2) 순서로 정렬. 형식이 다르면 맨 뒤."""
    numbers = re.findall(r"\d+", key)
    if numbers:
        return (0, [int(x) for x in numbers], key)
    return (1, [], key)


def load_filters(data):
    """filters 섹션을 읽어 {N: {Cross: 배열, X: 배열}}로 만든다.

    반환: (filters, filter_errors) - filter_errors는 {N: 실패 사유}
    """
    filters = {}
    filter_errors = {}

    raw_filters = data.get("filters") if isinstance(data, dict) else None
    if not isinstance(raw_filters, dict):
        print("✗ 'filters' 항목이 없거나 형식이 잘못되었습니다.")
        return filters, filter_errors

    for key in sorted(raw_filters, key=size_sort_key):
        match = re.fullmatch(r"size_(\d+)", key)
        if not match:
            print("✗ {}: 키 형식 오류 (size_N 형식이 아님) -> 건너뜀".format(key))
            continue
        n = int(match.group(1))

        entry = raw_filters[key]
        if not isinstance(entry, dict):
            filter_errors[n] = "{} 필터 형식 오류".format(key)
            print("✗ {}: 필터 형식 오류".format(key))
            continue

        pair = {}
        problems = []
        for filter_key, matrix in entry.items():
            label = normalize_label(filter_key)
            if label is None:
                problems.append("알 수 없는 필터 키 '{}'".format(filter_key))
                continue
            _, error = validate_matrix(matrix, n)
            if error:
                problems.append("{} 필터: {}".format(label, error))
                continue
            pair[label] = to_float_matrix(matrix)

        missing = [label for label in (CROSS, X) if label not in pair]
        if missing:
            reason = "{} 필터 로드 실패 ({})".format(
                key, "; ".join(problems) or "누락: " + ", ".join(missing))
            filter_errors[n] = reason
            print("✗ " + reason)
            continue

        filters[n] = pair
        print("✓ {:<8} 필터 로드 완료 (Cross, X)".format(key))
        for problem in problems:
            print("  (경고) " + problem)

    return filters, filter_errors


def evaluate_case(key, case, filters, filter_errors):
    """패턴 1개를 검증/판정한다. 문제가 있어도 예외 없이 결과 dict를 반환한다."""
    result = {"key": key, "cross": None, "x": None, "verdict": None,
              "expected": None, "passed": False, "reason": None}

    match = re.fullmatch(r"size_(\d+)_(\d+)", key)
    if not match:
        result["reason"] = "키 형식 오류 (size_N_idx 형식이 아님)"
        return result
    n = int(match.group(1))

    if n not in filters:
        result["reason"] = filter_errors.get(n, "size_{} 필터가 없습니다".format(n))
        return result

    if not isinstance(case, dict) or "input" not in case:
        result["reason"] = "스키마 오류: 'input' 항목이 없습니다"
        return result

    _, error = validate_matrix(case["input"], n)
    if error:
        result["reason"] = error
        return result

    pattern = to_float_matrix(case["input"])
    result["cross"] = mac(pattern, filters[n][CROSS])
    result["x"] = mac(pattern, filters[n][X])
    result["verdict"] = judge(result["cross"], result["x"], CROSS, X)

    raw_expected = case.get("expected")
    result["expected"] = normalize_label(raw_expected)
    if result["expected"] is None:
        result["reason"] = "라벨 오류: expected 값 {!r}를 해석할 수 없습니다".format(raw_expected)
        return result

    if result["verdict"] == UNDECIDED:
        diff = abs(result["cross"] - result["x"])
        result["reason"] = "동점(UNDECIDED) 처리 규칙에 따라 FAIL (|Cross-X| = {:.2e} < {:g})".format(
            diff, EPSILON)
        return result

    if result["verdict"] != result["expected"]:
        result["reason"] = "판정 불일치 (판정 {}, expected {})".format(
            result["verdict"], result["expected"])
        return result

    result["passed"] = True
    return result


def print_case(result):
    print("--- {} ---".format(result["key"]))
    if result["cross"] is not None:
        print("Cross 점수: " + format_number(result["cross"]))
        print("X 점수: " + format_number(result["x"]))

    verdict = result["verdict"] or "-"
    expected = result["expected"] or "-"
    status = "PASS" if result["passed"] else "FAIL"
    print("판정: {} | expected: {} | {}".format(verdict, expected, status))
    if not result["passed"]:
        print("사유: " + result["reason"])


def analyze_patterns(data, filters, filter_errors):
    raw_patterns = data.get("patterns") if isinstance(data, dict) else None
    if not isinstance(raw_patterns, dict):
        print("✗ 'patterns' 항목이 없거나 형식이 잘못되었습니다.")
        return []

    results = []
    for key in sorted(raw_patterns, key=size_sort_key):
        try:
            result = evaluate_case(key, raw_patterns[key], filters, filter_errors)
        except Exception as error:  # 어떤 케이스도 프로그램 전체를 멈추지 않게 한다
            result = {"key": key, "cross": None, "x": None, "verdict": None,
                      "expected": None, "passed": False,
                      "reason": "예기치 않은 오류: {}".format(error)}
        print_case(result)
        results.append(result)
    return results


def run_performance(filters):
    """크기별 MAC 시간 측정. data.json에 없는 크기(3x3 등)는 생성기로 만든다."""
    rows_2d = []
    rows_1d = []
    for n in PERF_SIZES:
        pattern = generate_cross(n)
        filt = filters[n][CROSS] if n in filters else generate_cross(n)
        rows_2d.append((n, measure_avg_ms(mac, pattern, filt)))

        # 변환 시간은 제외하고 연산 구간만 비교
        pattern_flat = flatten(pattern)
        filter_flat = flatten(filt)
        rows_1d.append((n, measure_avg_ms(mac_flat, pattern_flat, filter_flat)))

    print_perf_table(rows_2d)
    print()
    print_compare_table(rows_2d, rows_1d)


def print_compare_table(rows_2d, rows_1d):
    """[보너스] 같은 입력에 대한 2D / 1D MAC 시간 비교 표."""
    print("[보너스] 2차원 vs 1차원 배열 MAC 비교 (동일 입력, 평균/{}회)".format(REPEAT))
    table = []
    for (n, t2), (_, t1) in zip(rows_2d, rows_1d):
        ratio = t1 / t2 if t2 > 0 else float("nan")
        table.append(["{0}x{0}".format(n), "{:.5f}".format(t2), "{:.5f}".format(t1),
                      "{:.2f}".format(ratio)])
    print_table(["크기", "2D(ms)", "1D(ms)", "1D/2D"], [8, 14, 14, 10], table)


def print_summary(results):
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed

    print("총 테스트: {}개".format(total))
    print("통과: {}개".format(passed))
    print("실패: {}개".format(failed))
    if failed:
        print()
        print("실패 케이스:")
        for r in results:
            if not r["passed"]:
                print("- {}: {}".format(r["key"], r["reason"]))


def run_json_mode(path):
    data, error = load_json(path)
    if error:
        print()
        print("✗ " + error)
        print("data.json을 main.py와 같은 폴더에 두고 다시 실행하세요.")
        return

    print_section("[1] 필터 로드")
    filters, filter_errors = load_filters(data)

    print_section("[2] 패턴 분석 (라벨 정규화 적용)")
    results = analyze_patterns(data, filters, filter_errors)

    print_section("[3] 성능 분석 (평균/{}회)".format(REPEAT))
    run_performance(filters)

    print_section("[4] 결과 요약")
    print_summary(results)


# ---------------------------------------------------------------------------
# 모드 3: [보너스] 패턴 생성기
# ---------------------------------------------------------------------------
MIN_GENERATOR_SIZE = 3
MAX_GENERATOR_SIZE = 101
PRINT_LIMIT = 25  # 이보다 큰 패턴은 화면 출력을 생략한다


def read_size():
    """생성할 크기 N을 입력받는다. 정수가 아니거나 범위를 벗어나면 다시 입력받는다."""
    while True:
        text = read_line("크기 N 입력 ({}~{}): ".format(MIN_GENERATOR_SIZE, MAX_GENERATOR_SIZE)).strip()
        try:
            n = int(text)
        except ValueError:
            print("  입력 형식 오류: 정수를 입력하세요.")
            continue
        if not MIN_GENERATOR_SIZE <= n <= MAX_GENERATOR_SIZE:
            print("  입력 범위 오류: {} 이상 {} 이하의 정수를 입력하세요.".format(
                MIN_GENERATOR_SIZE, MAX_GENERATOR_SIZE))
            continue
        return n


def run_generator_mode():
    print_section("[1] 크기 입력")
    n = read_size()
    if n % 2 == 0:
        print("참고: 짝수 크기는 정중앙 칸이 없어 십자가가 가운데에서 한 칸 치우칩니다.")

    patterns = {label: GENERATORS[label](n) for label in (CROSS, X)}

    print_section("[2] 생성된 패턴 ({0}x{0})".format(n))
    for label in (CROSS, X):
        print("{} 패턴:".format(label))
        if n <= PRINT_LIMIT:
            print_matrix(patterns[label])
        else:
            print("  (크기가 {}보다 커서 출력을 생략합니다)".format(PRINT_LIMIT))

    # 생성한 Cross / X 패턴을 그대로 필터로도 사용해 판정이 맞는지 확인한다
    print_section("[3] MAC 판정 (생성 패턴을 필터로 사용)")
    for label in (CROSS, X):
        cross_score = mac(patterns[label], patterns[CROSS])
        x_score = mac(patterns[label], patterns[X])
        verdict = judge(cross_score, x_score, CROSS, X)
        print("--- {} 패턴 ---".format(label))
        print("Cross 점수: " + format_number(cross_score))
        print("X 점수: " + format_number(x_score))
        print("판정: {} | expected: {} | {}".format(
            verdict, label, "PASS" if verdict == label else "FAIL"))

    print_section("[4] 성능 분석 (평균/{}회)".format(REPEAT))
    pattern = patterns[CROSS]
    filt = patterns[CROSS]
    rows_2d = [(n, measure_avg_ms(mac, pattern, filt))]
    rows_1d = [(n, measure_avg_ms(mac_flat, flatten(pattern), flatten(filt)))]
    print_perf_table(rows_2d)
    print()
    print_compare_table(rows_2d, rows_1d)


# ---------------------------------------------------------------------------
# 진입점
# ---------------------------------------------------------------------------
def main():
    # 콘솔/리다이렉트 환경에서 한글·기호 출력이 깨지지 않도록 설정
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    print("=== Mini NPU Simulator ===")
    print()
    print("[모드 선택]")
    print("1. 사용자 입력 (3x3)")
    print("2. data.json 분석")
    print("3. 패턴 생성기 (보너스)")

    while True:
        choice = read_line("선택: ").strip()
        if choice == "1":
            run_user_mode()
            break
        if choice == "2":
            # 인자로 다른 JSON 경로를 주면 그 파일을 분석한다 (오류 케이스 테스트용)
            path = sys.argv[1] if len(sys.argv) > 1 else DATA_PATH
            run_json_mode(path)
            break
        if choice == "3":
            run_generator_mode()
            break
        print("1, 2, 3 중 하나를 입력하세요.")


if __name__ == "__main__":
    main()
