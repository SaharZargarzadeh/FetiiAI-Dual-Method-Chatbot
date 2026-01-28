
import re
from data_utils import last_n_days, any_passenger_age_cohort, downtown_box_mask

def answer_moody_last_month(trip):
    window = last_n_days(trip, 30)
    mask = window['Drop Off Address'].str.contains('moody center', case=False, na=False)
    count = int(mask.sum())
    return {"answer": f"{count} groups went to Moody Center in the last 30 days.", "table": window[mask]}

def answer_top_drop_18_24_sat_night(trip, ridermap, demo):
    t18 = any_passenger_age_cohort(trip, ridermap, demo, 18, 24)
    sat = t18[(t18['dow']=='Saturday') & (t18['hour'].between(20,23))]
    counts = sat['Drop Off Address'].value_counts().reset_index()
    counts.columns = ['Drop Off Address','rides']
    return {"answer": "Top drop-off spots for 18–24 on Saturday nights (20:00–23:00):", "table": counts}

def answer_large_groups_downtown_hours(trip):
    mask = (trip['Total Passengers']>=6) & downtown_box_mask(trip)
    grp = trip[mask].groupby('hour').size().reset_index(name='rides').sort_values('hour')
    peak = grp.sort_values('rides', ascending=False).head(3)['hour'].tolist()
    pretty_peak = ", ".join(f"{h:02d}:00" for h in peak)
    return {"answer": f"Large (6+) downtown rides peak around: {pretty_peak}.", "table": grp}

def generic_loc_count_last30(trip, location_text):
    window = last_n_days(trip, 30)
    mask = window['Drop Off Address'].str.contains(location_text, case=False, na=False)
    count = int(mask.sum())
    return {"answer": f"{count} groups went to '{location_text}' in the last 30 days.", "table": window[mask]}

def rule_based_router(trip, ridermap, demo, question:str):
    q = question.lower().strip()
    if ('moody center' in q) and ('how many' in q or 'count' in q):
        return answer_moody_last_month(trip)
    if ('top' in q and 'drop' in q and ('18-24' in q or '18–24' in q) and 'saturday' in q):
        return answer_top_drop_18_24_sat_night(trip, ridermap, demo)
    if (('when' in q or 'what time' in q) and ('large' in q or '6+' in q) and 'downtown' in q):
        return answer_large_groups_downtown_hours(trip)
    m = re.search(r'how many .* to (.+?) (?:last month|last 30 days)', q)
    if m:
        return generic_loc_count_last30(trip, m.group(1))
    return None

def gpt_intent_assist(openai_client, question:str):
    SYSTEM = "Map user questions about rideshare analytics to intents: moody_last_month | top_drop_18_24_sat_night | large_groups_downtown_hours | loc_last_30. Respond JSON {intent:..., location:...optional}."
    msg = [{"role":"system","content":SYSTEM},{"role":"user","content":question}]
    chat = openai_client.chat.completions.create(model="gpt-4o-mini", messages=msg, temperature=0)
    return chat.choices[0].message.content.strip()
