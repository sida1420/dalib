# AC-ACO / MRP Wireless Sensor Network Simulator

Mô phỏng mạng cảm biến không dây (Wireless Sensor Network — WSN) với bốn mode định tuyến chính: **AC-ACO Direct**, **Pure MRP**, **Hybrid AC-ACO + MRP**, và project adaptation **AC-ACO Direct-first + MRP fallback + Top-K**. Project giữ nguyên hành vi Baseline/Greedy và Pure MRP khi các adaptation tùy chọn bị tắt để kết quả cùng seed có thể tái lập.

Mỗi round mô phỏng tạo lưu lượng, tính năng lượng truyền/nhận vật lý, cập nhật năng lượng còn lại của từng sensor, và dừng theo điều kiện đã chọn. Final evaluation dùng cấu hình, scenario và seed đã đóng băng; diagnostic chỉ dùng để kiểm tra implementation, route, năng lượng hoặc runtime.

## Quick Start

Chạy các lệnh từ thư mục gốc repository.

```powershell
# 1) Dùng Python 3.10 (môi trường đã được kiểm thử)
python --version

# 2) Repository hiện không có requirements.txt/pyproject.toml.
# Cài các package được import bởi code và test:
python -m pip install numpy pandas matplotlib pytest

# 3) Chạy regression tests
python -m pytest -q

# 4) Smoke test 10 round của đúng một diagnostic method (Direct)
python run.py --mode direct --safety-max-rounds 10
```

Lần kiểm thử gần nhất của implementation hiện tại có **405 tests passed**. Hãy chạy test trước khi làm final evaluation.

## Chạy chương trình

### Chạy nhanh một phương pháp

Mỗi lệnh dưới đây chỉ chạy **một** phương pháp, lưu compact result rồi thoát process. Dùng `--native-case` khi muốn chạy trên map AC-ACO gốc (`map.pkl`); không cần `scenario` hoặc `anchor` cho case này.

```powershell
# Baseline: AC-ACO + Greedy
python run.py --mode direct --native-case --seed 33001

# Fair network-wide Pure MRP: MRP clustering + MRP routing
python run.py --mode mrp_pure --native-case --seed 33001

# Hybrid: AC-ACO clustering + MRP routing
python run.py --mode mrp_hybrid --native-case --seed 33001

# Project adaptation: Direct-first; only out-of-range CHs run MRP + Top-K
python run.py --mode direct_mrp_topk --native-case --seed 33001 --ttl 3 --phase3-top-k 3

# Optional physical control-ant accounting (100 bits is a project assumption)
python run.py --mode mrp_hybrid --native-case --seed 33001 --ttl 3 --phase3-top-k 3 --charge-ant-energy --ant-control-packet-bits 100

# Experimental: AC-ACO + MRP Phase II + lifetime-aware joint route selection
python run.py --mode mrp_hybrid_lifetime --native-case --seed 33001
```

Runner in trực tiếp và flush sau **mọi round thành công**, không chạy ngầm:

```text
round,alive_nodes,round_energy
1,100,0.0260399888722
2,100,0.0260981352354
```

Ba cột này đồng thời được ghi vào `round_state.csv` ở thư mục gốc. File được tạo lại cho mỗi lần chạy để không trộn seed/mode cũ. Có thể chọn đường dẫn khác:

```powershell
python run.py --mode mrp_hybrid --native-case --round-state-file run_outputs/hybrid-round-state.csv
```

JSON diagnostic cũ không bị ghi đè; lần chạy lại tự dùng hậu tố `_run2`, `_run3`, ... nên các lệnh trong README có thể chạy lại trực tiếp.

`MRP_SEARCH_FAILURE` là một lần tìm kiếm ngẫu nhiên không thấy route, chưa chứng minh mạng đã mất kết nối vật lý. Manual runner mặc định thử lại tối đa 3 lần trong cùng round. Khi ant-energy tắt, retry giữ behavior năng lượng cũ; khi bật, những control hop đã truyền vẫn bị charge kể cả discovery thất bại. Có thể đổi giới hạn hoặc khôi phục hành vi dừng ngay trước đây:

```powershell
python run.py --mode mrp_hybrid --native-case --mrp-search-retries 5
python run.py --mode mrp_hybrid --native-case --mrp-search-retries 0
```

Với scenario calibration, bỏ `--native-case` và truyền scenario/anchor thật:

```powershell
python run.py --mode mrp_pure --seed 33001 --scenario calibration-v1-c1 --anchor 49
```

Xem toàn bộ tùy chọn CLI:

```powershell
python run.py --help
```

### Tùy chọn charge năng lượng control ant

Mặc định, SANT/BANT/AANT giữ behavior lịch sử và không làm giảm physical battery. Muốn bật radio Tx/Rx accounting phải truyền đồng thời `--charge-ant-energy` và `--ant-control-packet-bits` dương. Paper yêu cầu cập nhật residual energy khi ant di chuyển nhưng không chỉ rõ kích thước packet theo bit, vì vậy runner không hard-code giá trị này. Convention control packet hiện có của project là 100 bit; các mức 50/100/200 bit chỉ là giả định sensitivity, không phải hằng số của MRP paper.

AANT trong implementation là nhánh forwarding của cùng một SANT, không phải packet thứ hai; một AANT hop chỉ bị charge một lần. Các Tx/Rx diagnostic count là sensor-battery radio operations: Sink không có residual-battery entry nên Sink-side operations không được tính vào count hay energy.

`run.py` là runner chẩn đoán thủ công: **mỗi lệnh chỉ chạy đúng một phương pháp** rồi lưu kết quả và thoát process. Không có lệnh nào tự động chạy tiếp hai mode còn lại, nên mỗi lần chạy được cách ly bộ nhớ bởi process riêng.

### Cấu hình TTL của MRP

`TTL` là giới hạn số sensor-to-sensor forward mà SANT được phép dùng trong Phase II trước khi tới Sink. TTL lớn hơn có thể tìm được route dài hơn, nhưng thường tăng số hop, thời gian chạy và chi phí search.

TTL chuẩn hiện có trong snapshot là **3**:

```text
real_calibration_v2/calibration_outputs/selected_config.json
    parameter_snapshot["mrp.ttl"] = 3
```

`run.py` hỗ trợ override TTL chỉ trong runtime cho các mode định tuyến MRP. Giá trị hiệu lực được ghi vào JSON output; file cấu hình chuẩn không bị sửa:

```powershell
python run.py --mode mrp_pure --ttl 5
python run.py --mode mrp_hybrid --ttl 5
python run.py --mode mrp_hybrid_lifetime --ttl 5
```

Không truyền `--ttl` thì runner tiếp tục dùng TTL từ snapshot chuẩn. `direct` không dùng MRP và sẽ từ chối cờ này.

Để thử TTL mà không thay đổi cấu hình chuẩn, dùng TTL sweep diagnostic hiện có:

```powershell
# Chạy hoặc tiếp tục sweep TTL 3, 4, 5, 6, 7, 8 cho mrp_pure và mrp_hybrid.
# Mỗi trial dùng process riêng, seed 33001 và original AC-ACO native case.
python run_ttl_lifetime_sweep.py
```

Sweep chỉ override `mrp.ttl` trong từng trial; nó không ghi đè `selected_config.json`, không dùng reserved seed `22001..22030`, và không ghi vào `final_evaluation_outputs/`. Kết quả nằm ở:

```text
diagnostic_ttl_lifetime_sweep/sweep_results.json
diagnostic_ttl_lifetime_sweep/trials/ttl_<n>/
```

Kết quả diagnostic hiện tại (một seed, không phải kết luận final): Pure MRP tốt nhất ở TTL 7, Hybrid tốt nhất ở TTL 4, và TTL chung tốt nhất theo tiêu chí `min(Pure lifetime, Hybrid lifetime)` là TTL 5. Giá trị này chỉ là khuyến nghị để review; **không tự sửa TTL chuẩn**.

Nếu muốn thay TTL cho final/scientific protocol, không chỉnh tay JSON rồi chạy final evaluation. Cần thực hiện lại calibration/provenance review, cập nhật snapshot có kiểm soát và tạo scientific freeze mới trước khi dùng reserved evaluation seeds.

### Direct

```powershell
python run.py --mode direct
```

Direct dùng Baseline hiện tại: AC-ACO chọn CH, sau đó Greedy/direct routing tạo route cuối cùng và energy model hiện có charge physical energy.

### Pure MRP — fair network-wide

```powershell
python run.py --mode mrp_pure
```

`mrp_pure` là `NETWORK_WIDE_MRP_ADAPTATION`: MRP chọn đúng N CH trên toàn live network, mỗi CH dùng Phase II/III chung với Hybrid, rồi tất cả N flow được tính bằng multi-flow energy ledger. Đây là adaptation phục vụ so sánh workload công bằng, không phải reproduction nguyên văn của paper.

Paper/event reference vẫn có sẵn và không bị xóa:

```powershell
python run.py --mode mrp_event
```

### Hybrid AC-ACO + MRP

```powershell
python run.py --mode mrp_hybrid
```

Ví dụ đầy đủ:

```powershell
python run.py --mode mrp_hybrid --seed 33001 --scenario calibration-v1-c1 --anchor 49
```

Aliases dễ đọc cũng được hỗ trợ: `pure`, `"mrp pure"`, `hybrid`, `"mrp hybrid"`, `event`, và `paper`. Để so sánh thủ công, chạy lần lượt `direct`, chờ process thoát, rồi `mrp_pure`, sau đó `mrp_hybrid`; không chạy chúng song song.

Runner này tiếp tục sau FND (First Node Death). Điều kiện dừng chính là traffic bắt buộc của mode không còn route hợp lệ tới Sink; lỗi search ngẫu nhiên chỉ kết thúc sau khi dùng hết `--mrp-search-retries`. `--safety-max-rounds` (mặc định `100000`) chỉ là guard vận hành, không phải metric lifetime. Kết quả JSON nằm dưới `run_outputs/direct/`, `run_outputs/mrp_pure/`, hoặc `run_outputs/mrp_hybrid/`; runner không ghi `final_evaluation_outputs/` và từ chối seed final reserved `22001..22030`.

## 1. Tổng quan

Project có bốn mode chính:

- **Baseline**: AC-ACO chọn cluster head (CH), sau đó dùng Greedy routing hiện có.
- **Pure MRP fair**: MRP adaptation chọn N CH trên toàn live network; Phase II/III chọn một route cho từng CH.
- **Hybrid AC-ACO + MRP**: AC-ACO chọn nhiều CH; mỗi CH dùng MRP độc lập để chọn route tới Sink.
- **Direct-first + MRP fallback + Top-K**: AC-ACO chọn CH; CH trong radio range truyền thẳng, CH ngoài range mới chạy MRP Phase II/III và optional Top-K.

Event-scoped Pure MRP gốc vẫn được giữ riêng dưới mode `mrp_event` làm `PAPER_EVENT_SCOPED_MRP_REFERENCE`.

```text
BASELINE:  AC-ACO chọn CH -> Greedy routing -> physical energy

PURE FAIR: Network-wide MRP adaptation chọn N CH -> Phase II cho từng CH
           -> Phase III cho từng CH -> N flows -> multi-flow physical energy

HYBRID:    AC-ACO chọn N CH -> Phase II cho từng CH -> Phase III cho từng CH
           -> N CH-to-Sink flows -> multi-flow physical energy

DIRECT+MRP: AC-ACO chọn N CH -> kiểm tra direct trước Phase II
            -> direct nếu tới Sink được, ngược lại MRP fallback + Top-K
```

`Sink` là trạm gốc. `CH` (cluster head) thu nhận/chuyển tiếp lưu lượng của cụm. `pheromone` là thông tin định hướng đường đi do MRP/ACO sử dụng; nó không phải năng lượng vật lý.

## 2. Bốn mode định tuyến chính

### Baseline

1. AC-ACO chọn các cluster head.
2. Greedy routing hiện có tạo topology định tuyến cuối cùng.
3. Energy model hiện có tính năng lượng truyền/nhận vật lý.

Baseline **không** dùng MRP để chọn route cuối cùng.

### Pure MRP fair: network-wide adaptation

Mode CLI `mrp_pure` dùng toàn bộ live network:

1. Adaptation Phase I tính score từ residual energy và live-neighbor count; không tạo signal/RSS giả.
2. Policy deterministic ưu tiên phủ các live sensor chưa được phủ, rồi score, rồi sensor ID để phá tie; chọn đúng target N từ `ACACOParameters.ch_proportion`.
3. Mọi live non-CH sensor được gán cho nearest selected CH giống Direct/Hybrid.
4. Common MRP Phase II và Phase III chạy độc lập cho từng CH.
5. Tất cả N selected flow dùng chung validated multi-flow energy ledger và physical residual state.

Phần residual-energy exponent và neighbor-count exponent xuất phát từ cấu trúc Eq.24. Coverage policy, việc bỏ event-only `SE_i`, tie-break và chọn nhiều CH là `NETWORK_WIDE_MULTI_CH_ADAPTATION_POLICY`, không phải claim từ paper.

### Paper/event Pure MRP reference

Mode `mrp_event` giữ nguyên pipeline lịch sử: event → Phase I Eq.24/Eq.25 chọn một event CH → Phase II → Phase III → một route tới Sink. Traffic chỉ thuộc event. Dùng mode này để kiểm tra reference behavior, không trộn nó vào comparison của các mode workload-fair.

### Hybrid AC-ACO + MRP

AC-ACO chọn **N** CH. Với **mỗi CH độc lập**:

1. MRP Phase II khám phá nhiều route CH-to-Sink.
2. MRP Phase III chọn đúng một route cho CH đó.
3. Tất cả N route được truyền như N flow độc lập tới Sink.

Hybrid là **multi-flow**, không phải một global single-parent tree. Các flow được phép dùng chung physical relay sensor, kể cả khi relay có next hop khác nhau theo từng flow:

```text
Flow CH84: 84 -> 66 -> Sink
Flow CH10: 10 -> 84 -> 23 -> Sink
```

Sensor `84` tham gia cả hai flow là hợp lệ. Năng lượng của mọi transmission/receive của nó được cộng vào **một** trạng thái residual-energy vật lý; không tạo bản sao sensor và không ép một parent toàn cục.

### Direct-first + MRP fallback + Top-K

Mode CLI `direct_mrp_topk` là project adaptation, không phải Original/Pure MRP:

1. AC-ACO chọn N CH như Hybrid.
2. Với từng CH, kiểm tra direct connectivity trước khi tạo SANT.
3. CH tới Sink được truyền trực tiếp bằng cùng physical data-energy ledger của Direct.
4. Chỉ CH ngoài range mới chạy MRP Phase II, existing Phase III fitness, Top-K pruning và probabilistic selection.
5. Pure MRP không nhận Top-K và Hybrid hiện có không bị biến thành Direct-first.

## 3. Cấu trúc project

```text
src/
  mrp/                         # Event Phase I, network-wide adaptation, common Phase II/III
  ac_aco_mrp/                  # Hybrid, AC-ACO Phase I, multi-flow plan và energy ledger
  experiments/                 # Fair Pure runner, harness, metrics, calibration/freeze
  evaluate.py                  # Primitive radio/energy và Greedy topology hiện có
  AC_ACO.py                    # Legacy implementation

tests/                         # Regression tests cho MRP, Hybrid và experiment harness
diagnostic_three_mode/         # Script diagnostic 3 mode, seed không reserved
run_checkpoint10.py            # Entry point final evaluation đã đóng băng
real_calibration_v2/           # Selected config, scenario manifest và Checkpoint-9D provenance
map.pkl                        # Network/map đầu vào hiện có
```

## 4. Yêu cầu môi trường

- **Python 3.10** là môi trường hiện được kiểm thử.
- Repository không có `requirements.txt`, `pyproject.toml` hay lockfile dependency.
- Import scan xác nhận `numpy`; test dùng `pytest`. `pandas` được legacy `src/AC_ACO.py` import; `matplotlib` được dùng cho visualization.

```powershell
python -m pip install numpy pandas matplotlib pytest
```

## 5. Testing

```powershell
python -m pytest -q
```

Nếu test fail, hãy chạy lệnh từ repository root, kiểm tra đúng Python environment và dependencies. Không tiếp tục final evaluation khi test/freeze preflight đang fail.

## 6. Chạy thử 1 trial

> Không dùng seed final `22001..22030` để thử nhanh. Dùng seed diagnostic **33001**.

Runner hỗ trợ đúng một mode mỗi lần gọi. Nó chạy tiếp sau FND và dừng tự nhiên khi mode đó không còn hoàn thành traffic bắt buộc tới Sink. Vì thế đây là route-exhaustion diagnostic, không phải frozen final protocol FND/5000.

```powershell
python run.py --mode direct
python run.py --mode mrp_pure --seed 33001 --scenario calibration-v1-c1 --anchor 49
python run.py --mode mrp_hybrid --seed 33001 --scenario calibration-v1-c1 --anchor 49
python run.py --mode direct_mrp_topk --native-case --seed 33001 --ttl 3 --phase3-top-k 3
python run.py --mode mrp_event --seed 33001 --scenario calibration-v1-c1 --anchor 49
```

Bốn mode workload-fair chính là `direct`, `mrp_pure`, `mrp_hybrid`, và `direct_mrp_topk`. Các alias `pure`, `"mrp pure"`, `hybrid`, `"mrp hybrid"` vẫn được hỗ trợ. `mrp_event` / `event` / `paper` là reference mode. Mỗi lệnh lưu compact result trong `run_outputs/<mode>/` và process thoát; không có mode nào tự khởi động mode kế tiếp.

`calibration-v1-c1` (anchor 49) là scenario không reserved. Fair `mrp_pure` chỉ dùng live hop snapshot của scenario; event membership/signal chỉ có ý nghĩa với `mrp_event`. Diagnostic không ghi `final_evaluation_outputs/`; đây là quan sát kỹ thuật, **không phải** kết luận khoa học hay bảng xếp hạng mode.

## 7. Diagnostic và final evaluation

### Diagnostic

Mục đích: kiểm tra route, energy, lỗi pheromone/routing, state isolation, thời gian chạy và bộ nhớ. Có thể dùng seed như `33001`; kết quả không phải final evidence.

### Final evaluation lịch sử — hiện đang stale

Checkpoint-9D freeze `411856bad35fbb8e` mô tả Pure MRP event-scoped, nên không còn authorize implementation fair network-wide. Kế hoạch 180 trial dưới đây là protocol lịch sử, **không được chạy lại** trước khi có review calibration và freeze/provenance mới:

| Mode | Native trials |
|---|---:|
| Baseline | 30 |
| Pure MRP | 4 scenarios × 30 seeds = 120 |
| Hybrid | 30 |
| **Tổng** | **180** |

- Reserved evaluation seeds: `22001..22030`.
- Primary stop: **FND** (First Node Death — node đầu tiên chết).
- Round cap: **5000**.
- Không có FND sau 5000 rounds nghĩa là **censored**, không phải FND ở round 5000.

## 8. Chạy final evaluation

> **Cảnh báo:** Không chạy final evaluation lúc này. Network-wide Pure-MRP adaptation đã làm Checkpoint-9D freeze cũ stale; preflight hiện phải từ chối nó. Cần review việc recalibration và tạo freeze mới trước khi dùng reserved seeds.

```powershell
python run_checkpoint10.py
```

`run_checkpoint10.py` vẫn là entry point lịch sử nhưng hiện **không được authorize**; source-identity preflight phát hiện freeze mismatch. Không chỉnh hoặc bỏ qua preflight để ép chạy.

## 9. Current Frozen Evaluation Configuration

Các ID dưới đây là **provenance/reproducibility identifiers**, không phải algorithm parameters:

| Mục | Giá trị |
|---|---|
| Scientific Freeze ID | `411856bad35fbb8e` — **STALE** |
| Scientific Freeze Hash | `411856bad35fbb8e724a4a2a81f6cb2121008f068af0fa837c2a18d7d74847e7` — **STALE** |
| Hybrid Implementation ID | `edefad02ed4242bd` |
| MRP parameter config ID | `62963da9e200e409` |

Freeze/provenance lịch sử nằm ở `real_calibration_v2/checkpoint_9d_freeze/scientific_freeze.json`. Selected parameter snapshot vẫn không đổi, nhưng việc có cần recalibrate cho source-CH/workload distribution mới phải được review; repository chưa tự quyết định và chưa tạo freeze mới.

## 10. Tóm tắt MRP parameters hiện tại

Đây là các giá trị đã xác minh từ `selected_config.json`:

| Parameter | Giá trị |
|---|---:|
| `alpha` | 2 |
| `beta` | 2 |
| `rho` | 0.2 |
| initial pheromone | 0.01 |
| AANT probability | 0.001 |
| `c0` | 1.0 |
| `c` | 0.005 |
| `c1` | 0.003696666640752299 |
| `lambda` | 9.04421850555005 |
| `num_sants` | 20 |
| `TTL` | 3 |
| pheromone lifecycle | `RESET_PER_DISCOVERY` |

## 11. Energy accounting và metrics

Mặc định `charge_ant_energy=False`, physical residual energy chỉ bị charge cho **final physical data communication** của round thành công. Không charge residual energy cho:

- AC-ACO candidate-fitness/Greedy oracle;
- `RouteQuality.f2` hoặc Phase-III fitness;
- route bị reject trong discovery.

Khi bật `charge_ant_energy`, mỗi SANT/AANT/BANT hop thực sự truyền được charge bằng cùng radio model. Search failure không rollback control energy đã tiêu thụ. Data energy và ant-control energy được báo riêng; tổng luôn thỏa:

```text
total_physical_energy = data_energy + total_ant_control_energy
```

Với Hybrid và Direct-first, energy ledger cộng mọi Tx/Rx của các selected CH flows vào cùng một sensor vật lý. Direct-first không tạo ant cho CH direct. Pure MRP tiếp tục dùng toàn bộ discovered routes trong Phase III và không dùng Top-K adaptation.

Các metric chính:

- **FND**: First Node Death.
- **physical energy**: tổng năng lượng communication thực tế đã charge.
- **residual energy**: năng lượng còn lại sau round/trial.
- **delivered payload count**: số payload được mô hình truyền thành công.
- **energy per delivered payload**: physical energy chuẩn hóa theo payload.

Bốn mode chính `direct`, `mrp_pure`, `mrp_hybrid`, `direct_mrp_topk` đều mô phỏng traffic của tất cả live node, nên raw energy có workload basis gần nhau hơn trước. `energy per delivered payload` vẫn là normalized diagnostic quan trọng. Mode reference `mrp_event` vẫn event-scoped; raw energy của nó không được so trực tiếp với các mode workload-fair.

## 12. Output files

Sau final evaluation thành công, `real_calibration_v2/final_evaluation_outputs/` được publish atomically với các file sau:

| File | Nội dung |
|---|---|
| `evaluation_manifest.json` | Freeze/config, scenario, seed và protocol provenance |
| `native_trial_metrics.csv` | Một dòng cho mỗi native trial terminal |
| `round_metrics.csv` | Metric theo round |
| `seed_level_metrics.csv` | Tổng hợp theo seed |
| `failures.csv` | Các round/trial failure, không bị bỏ qua |
| `summary.json`, `summary.csv` | Summary theo mode |
| `pure_mrp_scenario_summary.csv` | Summary Pure MRP theo event scenario |

Diagnostic script hiện có chủ yếu in thông tin route/energy ra terminal; nó không được coi là output khoa học final.

## 13. Troubleshooting

### Tests fail

Chạy lại từ repository root, kiểm tra `python --version` và cài đúng dependencies. Không chạy final evaluation cho tới khi test và freeze preflight pass.

### No MRP route

`Phase II produced no unique successful routes` có thể là kết quả routing thực sự của network state/seed/round, không nhất thiết là Python crash. Giữ lại failure record để phân tích; không tự động thay bằng Greedy route.

### Negative pheromone

Implementation hiện fail fast khi pheromone âm. Không clamp giá trị âm để “chạy tiếp”, vì điều đó thay đổi semantics và làm mất khả năng tái lập.

### Hybrid shared relay

Relay được dùng bởi nhiều CH flows là hợp lệ, kể cả khi next hop khác nhau. Không sửa bằng cách ép một global parent hoặc loại route chỉ vì shared relay.

### Process dùng nhiều bộ nhớ hoặc bị terminate

Final runner hiện stage canonical trial records và không giữ lịch sử của cả 180 trials trong RAM. Nếu process bị terminate, kiểm tra `checkpoint10_progress.json` và staging directory `.final_evaluation_outputs.staging.*`; review canonical trial state trước khi quyết định recovery. Không blind-restart; runner hiện không tự động resume để tránh duplicate observation.

## 14. Important Notes

- `src/AC_ACO.py` là legacy code và gọi `run()` ở module scope; import trực tiếp có thể thực thi legacy simulation và ghi legacy artifacts. New integrations không nên import file này tùy tiện.
- `src/experiments/baseline_adapter.py` giữ Baseline/Greedy semantics mà không import `AC_ACO.py`. Đây là chủ đích để bảo toàn experimental comparability.
- Không “clean up” legacy routing/energy semantics nếu chưa đánh giá tác động đến reproducibility và scientific freeze.

## 15. Reproducibility

Seed điều khiển random behavior; calibration seed và evaluation seed được tách riêng. Calibration scenarios và evaluation scenarios cũng là các artifact frozen riêng biệt.

Mọi thay đổi algorithm, parameter, traffic semantics hoặc Hybrid routing semantics có thể làm scientific freeze không còn hợp lệ. Khi đó cần cập nhật provenance/freeze trước khi chạy final evaluation; không tái sử dụng reserved results như thể implementation chưa đổi.
