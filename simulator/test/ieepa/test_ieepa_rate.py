import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from datetime import date
from simulator.sectionieepa_rate import compute_sectionieepa_duty
os.environ['DATABASE_DSN'] = 'postgresql://postgres:Xylx1.t123@34.129.224.77:5432/tariff-simulate'
res = compute_sectionieepa_duty(
    hts_number="7319.90.10.00",
    country_of_origin="AF",
    entry_date=date.fromisoformat("2025-10-20"),
    import_value=100000.0,
    melt_pour_origin_iso2="US",
    measurements={"steel_percentage": 0.2, "aluminum_percentage": 0.3},
)
print("applicable len", len(res.ch99_list))
for ch in res.ch99_list:
    print(ch.ch99_id, ch.alias, ch.general_rate, ch.ch99_description, ch.amount, ch.is_potential)