# UBI (Umbrella Before Impact) — climate-vulnerable commercial districts

Gangnam-gu and Chuncheon-si, 2025-07 to 2025-12. Model: log(daily sales) ~ month + weekday + holiday + heat wave + ln(1 + rain), OLS with Newey-West (lag 7).

## Layout
    project/
    ├── run_all.py
    ├── UBI/                          all source code (run in order 1-4)
    │   ├── ubi_model.py                  1. model fit, vulnerable industries
    │   ├── ubi_damage_priority.py        2. damage size, closed-form intervals, priority
    │   ├── ubi_dashboard_precompute.py   3. per day forecast damage (hybrid baseline)
    │   ├── ubi_build_dashboard.py        4. data.json, index.html
    │   └── ubi_dashboard_template.html
    ├── preprocessed_data/            inputs
    │   ├── SH_card_data1_preprocessed.csv   (not committed)
    │   ├── weather_event.csv
    │   └── forecast/short17_reg_{gangnam,chuncheon}_d1.csv
    └── outputs/                      results

## Run
    pip install -r requirements.txt
    python run_all.py                       # reads preprocessed_data/SH_card_data1_preprocessed.csv
    python run_all.py path/to/card.csv      # or pass a path
    python run_all.py --steps=4             # only some steps

## Slides
    [📎발표자료 보기](UBI_presentation.pdf)
    <img width="1012" height="448" alt="image" src="https://github.com/user-attachments/assets/dc7d1ba6-65c1-4db3-97f1-a6e8c4c3cab5" />

