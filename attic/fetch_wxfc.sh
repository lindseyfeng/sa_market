#!/bin/bash
cd /Users/lindseyf/sa_market
# Historical FORECAST archive: what the model predicted at the time, not reanalysis.
# Fixes the standing "weather is reanalysis => upper bound" caveat, and is the
# only legitimately forward-looking weather for h>1.
curl -sSL --retry 2 --max-time 900 -o wxfc_raw.json \
 "https://historical-forecast-api.open-meteo.com/v1/forecast?latitude=-34.93,-33.15,-37.70,-37.81&longitude=138.60,138.60,140.40,144.96&start_date=2018-01-01&end_date=2022-12-31&hourly=temperature_2m,wind_speed_100m,shortwave_radiation&timezone=Australia%2FAdelaide"
echo "WXFC DONE: $(wc -c < wxfc_raw.json) bytes"
python3 -c "
import json;d=json.load(open('wxfc_raw.json'))
ds=d if isinstance(d,list) else [d]
print('locations:',len(ds))
for i,x in enumerate(ds):
    h=x.get('hourly',{}); t=h.get('time',[])
    print(f'  loc{i}: lat={x.get(\"latitude\")} n={len(t)} {t[0] if t else \"-\"} -> {t[-1] if t else \"-\"}')
"
